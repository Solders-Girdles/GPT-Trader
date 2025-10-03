# State Manager Refactoring Plan

**Date**: 2025-10-02
**Target**: `src/bot_v2/state/state_manager.py`
**Current Size**: 715 lines
**Target Size**: ~400 lines (44% reduction)

## Problem Statement

StateManager's `__init__` method (75 lines, lines 106-180) mixes multiple responsibilities:
- Adapter bootstrapping (Redis, Postgres, S3)
- Adapter validation
- Repository initialization
- Policy/cache/metrics initialization

**Already Extracted (Good!):**
- ✅ `StateCacheManager` - Cache management logic
- ✅ `StatePerformanceMetrics` - Performance tracking
- ✅ `TierPromotionPolicy` - Promotion/demotion rules
- ✅ `StorageAdapterFactory` - Adapter creation
- ✅ Repository classes - RedisStateRepository, PostgresStateRepository, S3StateRepository

**Remaining Issues:**
1. `__init__` still manually creates/validates adapters (48 lines)
2. `__init__` directly instantiates repositories (11 lines)
3. No clear separation between initialization and runtime coordination

## Current Structure

### StateManager.__init__ (75 lines)

```python
def __init__(self, config, redis_adapter=None, postgres_adapter=None, s3_adapter=None):
    # Line 113-114: Initialize config and lock
    self.config = config or StateConfig()
    self._lock = threading.Lock()

    # Line 116-117: Create cache manager
    self._cache_manager = StateCacheManager(config=self.config)

    # Line 119-120: Create factory
    factory = StorageAdapterFactory()

    # Lines 122-154: Adapter creation/validation (48 lines!)
    if redis_adapter is None:
        self.redis_adapter = factory.create_redis_adapter(...)
    else:
        self.redis_adapter = redis_adapter

    if postgres_adapter is None:
        self.postgres_adapter = factory.create_postgres_adapter(...)
    else:
        self.postgres_adapter = factory.validate_postgres_adapter(postgres_adapter)

    if s3_adapter is None:
        self.s3_adapter = factory.create_s3_adapter(...)
    else:
        self.s3_adapter = factory.validate_s3_adapter(s3_adapter, bucket)

    # Lines 155-166: Repository initialization (11 lines)
    self._redis_repo = RedisStateRepository(...) if self.redis_adapter else None
    self._postgres_repo = PostgresStateRepository(...) if self.postgres_adapter else None
    self._s3_repo = S3StateRepository(...) if self.s3_adapter else None

    # Lines 168-176: Policy and metrics initialization
    self._promotion_policy = TierPromotionPolicy(...)
    self._metrics = StatePerformanceMetrics(...)
```

## Proposed Extraction

### Phase 1: Extract AdapterBootstrapper

**Location**: `src/bot_v2/state/adapter_bootstrapper.py`

**Responsibilities:**
- Create or validate adapters based on config
- Handle adapter fallback logic (provided vs. created)
- Encapsulate adapter initialization complexity

**API:**
```python
@dataclass
class BootstrappedAdapters:
    """Bundle of initialized storage adapters."""
    redis: RedisAdapter | None
    postgres: PostgresAdapter | None
    s3: S3Adapter | None


class AdapterBootstrapper:
    """Handles adapter creation and validation for StateManager."""

    def __init__(self, factory: StorageAdapterFactory):
        self.factory = factory

    def bootstrap(
        self,
        config: StateConfig,
        redis_adapter: RedisAdapter | None = None,
        postgres_adapter: PostgresAdapter | None = None,
        s3_adapter: S3Adapter | None = None,
    ) -> BootstrappedAdapters:
        """
        Bootstrap storage adapters with fallback to creation.

        Args:
            config: State configuration
            redis_adapter: Optional pre-configured Redis adapter
            postgres_adapter: Optional pre-configured Postgres adapter
            s3_adapter: Optional pre-configured S3 adapter

        Returns:
            BootstrappedAdapters with all adapters ready
        """
        return BootstrappedAdapters(
            redis=self._bootstrap_redis(config, redis_adapter),
            postgres=self._bootstrap_postgres(config, postgres_adapter),
            s3=self._bootstrap_s3(config, s3_adapter),
        )

    def _bootstrap_redis(
        self, config: StateConfig, adapter: RedisAdapter | None
    ) -> RedisAdapter | None:
        """Bootstrap Redis adapter."""
        if adapter is None:
            return self.factory.create_redis_adapter(
                host=config.redis_host,
                port=config.redis_port,
                db=config.redis_db,
            )
        return adapter

    def _bootstrap_postgres(
        self, config: StateConfig, adapter: PostgresAdapter | None
    ) -> PostgresAdapter | None:
        """Bootstrap and validate Postgres adapter."""
        if adapter is None:
            return self.factory.create_postgres_adapter(
                host=config.postgres_host,
                port=config.postgres_port,
                database=config.postgres_database,
                user=config.postgres_user,
                password=config.postgres_password,
            )
        # Validate provided adapter
        return self.factory.validate_postgres_adapter(adapter)

    def _bootstrap_s3(
        self, config: StateConfig, adapter: S3Adapter | None
    ) -> S3Adapter | None:
        """Bootstrap and validate S3 adapter."""
        if adapter is None:
            return self.factory.create_s3_adapter(
                region=config.s3_region,
                bucket=config.s3_bucket,
            )
        # Validate provided adapter
        return self.factory.validate_s3_adapter(adapter, config.s3_bucket)
```

**Lines extracted**: ~60 lines (from __init__)

---

### Phase 2: Extract RepositoryFactory

**Location**: `src/bot_v2/state/repository_factory.py`

**Responsibilities:**
- Create tier-specific repositories from adapters
- Handle null adapter cases gracefully
- Encapsulate repository initialization logic

**API:**
```python
class RepositoryFactory:
    """Creates tier-specific repositories from storage adapters."""

    @staticmethod
    def create_repositories(
        adapters: BootstrappedAdapters,
        config: StateConfig,
    ) -> StateRepositories:
        """
        Create tier-specific repositories from adapters.

        Args:
            adapters: Bootstrapped storage adapters
            config: State configuration

        Returns:
            StateRepositories bundle
        """
        redis_repo = (
            RedisStateRepository(adapters.redis, config.redis_ttl_seconds)
            if adapters.redis
            else None
        )

        postgres_repo = (
            PostgresStateRepository(adapters.postgres)
            if adapters.postgres
            else None
        )

        s3_repo = (
            S3StateRepository(adapters.s3, config.s3_bucket)
            if adapters.s3
            else None
        )

        return StateRepositories(
            redis=redis_repo,
            postgres=postgres_repo,
            s3=s3_repo,
        )
```

**Lines extracted**: ~20 lines (from __init__)

---

## Refactored StateManager

**Target**: ~400 lines (down from 715)

```python
class StateManager:
    """Central state coordination with multi-tier storage."""

    def __init__(
        self,
        config: StateConfig | None = None,
        redis_adapter: RedisAdapter | None = None,
        postgres_adapter: PostgresAdapter | None = None,
        s3_adapter: S3Adapter | None = None,
    ) -> None:
        self.config = config or StateConfig()
        self._lock = threading.Lock()

        # Bootstrap adapters
        factory = StorageAdapterFactory()
        bootstrapper = AdapterBootstrapper(factory)
        adapters = bootstrapper.bootstrap(
            self.config,
            redis_adapter=redis_adapter,
            postgres_adapter=postgres_adapter,
            s3_adapter=s3_adapter,
        )

        # Store adapters for access
        self.redis_adapter = adapters.redis
        self.postgres_adapter = adapters.postgres
        self.s3_adapter = adapters.s3

        # Create repositories
        repos = RepositoryFactory.create_repositories(adapters, self.config)
        self._redis_repo = repos.redis
        self._postgres_repo = repos.postgres
        self._s3_repo = repos.s3

        # Initialize components
        self._cache_manager = StateCacheManager(config=self.config)
        self._promotion_policy = TierPromotionPolicy(
            redis_repo=self._redis_repo,
            postgres_repo=self._postgres_repo,
            s3_repo=self._s3_repo,
        )
        self._metrics = StatePerformanceMetrics(
            enabled=self.config.enable_performance_tracking
        )

    # ... rest of methods unchanged ...
```

## Benefits

1. **Reduced Complexity**: __init__: 75 → ~35 lines (53% reduction)
2. **Single Responsibility**: Each component has one clear job
3. **Testability**: Can test adapter bootstrapping and repository creation independently
4. **Maintainability**: Easy to modify adapter initialization logic
5. **Backward Compatibility**: 100% API compatibility maintained

## Implementation Plan

1. **Phase 1**: Extract AdapterBootstrapper (~1.5h)
   - Create new file
   - Move adapter creation/validation logic
   - Write unit tests
   - Update StateManager to use bootstrapper

2. **Phase 2**: Extract RepositoryFactory (~30min)
   - Create repository factory
   - Move repository initialization
   - Write unit tests
   - Update StateManager

3. **Phase 3**: Integration Testing (~30min)
   - Run existing test suite
   - Verify backward compatibility
   - Test with null adapters

**Total Estimated Time**: 2.5 hours

## Success Criteria

- [x] StateManager.__init__ reduced to <40 lines ✅ (32 lines, 55% reduction from 71 lines)
- [x] All existing tests passing ✅ (663 tests pass)
- [x] New unit tests for components (10+ tests) ✅ (32 new tests: 19 + 13)
- [x] Backward compatibility maintained ✅ (all 631 original state tests pass)
- [x] No functionality changes ✅ (verified via test suite)

## Completion Status

**Status**: ✅ **COMPLETED** (2025-10-02)

### Actual Results

**Files Created:**
1. `src/bot_v2/state/adapter_bootstrapper.py` (139 lines)
   - BootstrappedAdapters dataclass
   - AdapterBootstrapper class with bootstrap logic
   - 19 unit tests in test_adapter_bootstrapper.py

2. `src/bot_v2/state/repository_factory.py` (64 lines)
   - RepositoryFactory static class
   - 13 unit tests in test_repository_factory.py

**Modified:**
- `src/bot_v2/state/state_manager.py`: 716 → 685 lines (31 lines / 4.3% reduction)
  - `__init__`: 71 → 32 lines (39 lines / 55% reduction)
  - Updated imports to include new modules

**Test Results:**
- Total tests: 663 (631 original + 32 new)
- All tests passing ✅
- Zero regressions
- 100% backward compatibility maintained

**Metrics:**
- StateManager.__init__ complexity reduced by 55%
- 32 new unit tests added
- Adapter bootstrapping logic fully extracted and testable
- Repository creation logic fully extracted and testable
- Constructor signature unchanged (perfect backward compatibility)

**Test Commands:**
```bash
# Run new module tests
pytest tests/unit/bot_v2/state/test_adapter_bootstrapper.py -v  # 19 tests
pytest tests/unit/bot_v2/state/test_repository_factory.py -v    # 13 tests

# Run full state test suite
pytest tests/unit/bot_v2/state/ -v  # 663 tests
```

**Git Reference:**
- Baseline SHA: f5ec84eb25442c0956439d39d117312f99bf503e
- Original __init__: Lines 106-176 (71 lines)
- Refactored __init__: Lines 103-141 (32 lines)

## Files to Create

**New files:**
- `src/bot_v2/state/adapter_bootstrapper.py`
- `src/bot_v2/state/repository_factory.py`
- `tests/unit/bot_v2/state/test_adapter_bootstrapper.py`
- `tests/unit/bot_v2/state/test_repository_factory.py`

**Modified files:**
- `src/bot_v2/state/state_manager.py`

**Test files to run:**
- `tests/unit/bot_v2/state/test_state_manager.py`
