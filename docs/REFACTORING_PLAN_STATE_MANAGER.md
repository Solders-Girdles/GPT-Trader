# State Manager Refactoring Plan

**File**: `src/bot_v2/state/state_manager.py`
**Current Size**: 690 lines
**Target**: Split into 4-5 focused modules (~150-200 lines each)

---

## Current Structure Analysis

### Responsibilities Identified

1. **Caching Logic** (~100 lines)
   - `_local_cache()`, `_metadata_cache()`, `_access_history()`
   - `_update_access_history()`, `_manage_cache_size()`
   - `_calculate_checksum()`

2. **Repository Operations** (~150 lines)
   - `_get_from_redis()`, `_get_from_postgres()`, `_get_from_s3()`
   - `_set_in_redis()`, `_set_in_postgres()`, `_set_in_s3()`

3. **State Operations** (~200 lines)
   - `get_state()`, `set_state()`, `delete_state()`
   - `get_keys_by_pattern()`

4. **Tier Management** (~50 lines)
   - `promote_to_hot()`, `demote_to_cold()`

5. **Batch Operations** (~150 lines)
   - `batch_delete_state()`, `batch_set_state()`

6. **Metrics & Stats** (~40 lines)
   - `get_storage_stats()`, `get_performance_metrics()`

---

## Proposed Module Structure

### 1. `state_cache.py` (Cache Management)
**Size**: ~120 lines
**Responsibility**: In-memory caching, access tracking, checksum calculation

```python
class StateCache:
    """Manages in-memory state caching with LRU eviction."""

    def __init__(self, max_size: int = 1000):
        self._local_cache: dict[str, Any] = {}
        self._metadata_cache: dict[str, StateMetadata] = {}
        self._access_history: dict[str, list[datetime]] = {}
        self._lock = threading.RLock()
        self._max_cache_size = max_size

    def get(self, key: str) -> Any | None:
        """Get value from cache, update access history."""

    def set(self, key: str, value: Any, metadata: StateMetadata) -> None:
        """Set value in cache, manage size."""

    def invalidate(self, key: str) -> None:
        """Remove key from cache."""

    def update_access_history(self, key: str) -> None:
        """Track access for promotion/demotion decisions."""

    def manage_size(self) -> None:
        """Evict least recently used entries if over limit."""

    @staticmethod
    def calculate_checksum(data: str) -> str:
        """Calculate checksum for data integrity."""
```

**Benefits**:
- Single Responsibility: Cache operations only
- Testable in isolation
- Reusable across different state managers

---

### 2. `state_repository_coordinator.py` (Repository Coordination)
**Size**: ~180 lines
**Responsibility**: Coordinate access across Redis/Postgres/S3

```python
class StateRepositoryCoordinator:
    """Coordinates state operations across storage tiers."""

    def __init__(
        self,
        redis_repo: RedisStateRepository | None,
        postgres_repo: PostgresStateRepository | None,
        s3_repo: S3StateRepository | None,
    ):
        self.redis = redis_repo
        self.postgres = postgres_repo
        self.s3 = s3_repo

    async def get_from_tier(self, key: str, tier: StateCategory) -> Any | None:
        """Get state from specific tier."""

    async def set_in_tier(
        self, key: str, value: str, tier: StateCategory, **kwargs
    ) -> bool:
        """Set state in specific tier."""

    async def delete_from_tier(self, key: str, tier: StateCategory) -> bool:
        """Delete state from specific tier."""

    async def get_keys_by_pattern(self, pattern: str) -> list[str]:
        """Get keys matching pattern from all tiers."""

    def get_repositories(self) -> StateRepositories:
        """Get direct repository access."""
```

**Benefits**:
- Encapsulates tier-specific logic
- Easy to mock for testing
- Clear separation from business logic

---

### 3. `state_tier_manager.py` (Tier Promotion/Demotion)
**Size**: ~100 lines
**Responsibility**: Handle data movement between tiers

```python
class StateTierManager:
    """Manages state promotion and demotion across storage tiers."""

    def __init__(
        self,
        coordinator: StateRepositoryCoordinator,
        cache: StateCache,
        policy: TierPromotionPolicy,
    ):
        self.coordinator = coordinator
        self.cache = cache
        self.policy = policy

    async def promote_to_hot(self, key: str) -> bool:
        """Promote state from warm/cold to hot tier."""

    async def demote_to_cold(self, key: str) -> bool:
        """Demote state from hot/warm to cold tier."""

    async def auto_promote_if_needed(self, key: str, access_count: int) -> bool:
        """Auto-promote based on access patterns."""

    def should_promote(self, key: str) -> bool:
        """Check if key meets promotion criteria."""
```

**Benefits**:
- Focused on tier movement logic
- Policy-driven decisions
- Easy to extend with new promotion strategies

---

### 4. `state_batch_operations.py` (Batch Operations)
**Size**: ~180 lines
**Responsibility**: Efficient bulk operations

```python
class StateBatchOperations:
    """Handles batch state operations for efficiency."""

    def __init__(
        self,
        coordinator: StateRepositoryCoordinator,
        cache: StateCache,
    ):
        self.coordinator = coordinator
        self.cache = cache

    async def batch_get(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple states efficiently."""

    async def batch_set(
        self,
        items: dict[str, Any],
        category: StateCategory,
        ttl_seconds: int | None = None,
    ) -> int:
        """Set multiple states efficiently."""

    async def batch_delete(self, keys: list[str]) -> int:
        """Delete multiple states efficiently."""

    async def batch_promote(self, keys: list[str]) -> int:
        """Promote multiple keys to hot tier."""
```

**Benefits**:
- Optimized for bulk operations
- Separate from single-item logic
- Easy to benchmark and optimize

---

### 5. `state_manager.py` (Main Orchestrator - REFACTORED)
**Size**: ~200 lines
**Responsibility**: High-level API, orchestrates components

```python
class StateManager:
    """
    High-level state management API.

    Orchestrates caching, tier coordination, and batch operations
    for optimal performance across Redis, PostgreSQL, and S3 storage.
    """

    def __init__(self, config: StateConfig):
        # Initialize components
        self._cache = StateCache(max_size=config.max_cache_size)
        self._coordinator = self._build_coordinator(config)
        self._tier_manager = StateTierManager(
            self._coordinator, self._cache, TierPromotionPolicy()
        )
        self._batch_ops = StateBatchOperations(self._coordinator, self._cache)
        self._metrics = StatePerformanceMetrics()

    async def get_state(self, key: str, auto_promote: bool = True) -> Any | None:
        """
        Get state from appropriate tier.

        1. Check local cache
        2. Query Redis → Postgres → S3 (waterfall)
        3. Auto-promote if access patterns warrant
        4. Update cache and metrics
        """
        # Delegate to components

    async def set_state(
        self,
        key: str,
        value: Any,
        category: StateCategory = StateCategory.HOT,
        ttl_seconds: int | None = None,
    ) -> bool:
        """Set state in specified tier."""
        # Delegate to coordinator

    async def delete_state(self, key: str) -> bool:
        """Delete state from all tiers."""
        # Delegate to coordinator

    # Delegation methods for tier management
    async def promote_to_hot(self, key: str) -> bool:
        return await self._tier_manager.promote_to_hot(key)

    async def demote_to_cold(self, key: str) -> bool:
        return await self._tier_manager.demote_to_cold(key)

    # Delegation methods for batch operations
    async def batch_set_state(self, items: dict[str, Any], **kwargs) -> int:
        return await self._batch_ops.batch_set(items, **kwargs)

    async def batch_delete_state(self, keys: list[str]) -> int:
        return await self._batch_ops.batch_delete(keys)

    # Metrics and stats
    async def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics from all tiers."""

    def get_performance_metrics(self) -> dict[str, Any]:
        return self._metrics.to_dict()

    def get_repositories(self) -> StateRepositories:
        return self._coordinator.get_repositories()

    def close(self) -> None:
        """Close all connections."""
```

**Benefits**:
- Clean, high-level API
- Delegates to specialized components
- Easy to understand and maintain
- Testable with mocked components

---

## Refactoring Strategy

### Phase 1: Extract Components (Bottom-Up)
1. **Create `state_cache.py`** (30 min)
   - Extract caching logic
   - Add tests

2. **Create `state_repository_coordinator.py`** (45 min)
   - Extract repository coordination
   - Add tests

3. **Create `state_tier_manager.py`** (30 min)
   - Extract tier promotion/demotion
   - Add tests

4. **Create `state_batch_operations.py`** (45 min)
   - Extract batch operations
   - Add tests

### Phase 2: Refactor Main Class (Top-Down)
5. **Refactor `state_manager.py`** (60 min)
   - Update to use new components
   - Simplify to orchestrator pattern
   - Update existing tests

---

## Testing Strategy

### Unit Tests (New)
- `tests/unit/bot_v2/state/test_state_cache.py`
- `tests/unit/bot_v2/state/test_state_repository_coordinator.py`
- `tests/unit/bot_v2/state/test_state_tier_manager.py`
- `tests/unit/bot_v2/state/test_state_batch_operations.py`

### Integration Tests (Update)
- Update existing `test_state_manager.py` to verify orchestration
- Ensure backward compatibility

---

## Benefits Summary

### Code Quality
- ✅ **Single Responsibility**: Each module has one clear purpose
- ✅ **Testability**: Components easy to test in isolation
- ✅ **Maintainability**: Easier to understand and modify
- ✅ **Reusability**: Components can be used independently

### Team Benefits
- ✅ **Faster Onboarding**: Clear module boundaries
- ✅ **Parallel Development**: Multiple devs can work on different components
- ✅ **Easier Debugging**: Smaller scope per module

### Performance
- ✅ **No Regression**: Same algorithms, better organization
- ✅ **Future Optimization**: Easier to optimize individual components

---

## Migration Path

### Backward Compatibility
- Keep `StateManager` public API unchanged
- Internal refactoring only
- All existing tests should pass without modification

### Rollout
1. Create new component modules
2. Update `StateManager` to use components internally
3. Run full test suite
4. Deploy with monitoring
5. Remove old code after validation

---

## Success Criteria

✅ **Code Metrics**:
- No file >300 lines
- All modules <200 lines
- Test coverage ≥85%

✅ **Functionality**:
- All existing tests pass
- No performance regression
- Backward compatible API

✅ **Quality**:
- Clear module boundaries
- No circular dependencies
- Type hints complete

---

## Estimated Effort

- **Phase 1 (Extract Components)**: 2.5 hours
- **Phase 2 (Refactor Main)**: 1 hour
- **Testing & Validation**: 1 hour
- **Total**: ~4.5 hours

---

## Next Steps

1. Review this plan with team
2. Create feature branch: `refactor/state-manager-decomposition`
3. Execute Phase 1 (bottom-up extraction)
4. Execute Phase 2 (top-down refactoring)
5. Code review and merge
