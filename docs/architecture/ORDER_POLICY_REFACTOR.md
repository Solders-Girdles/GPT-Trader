# OrderPolicy Refactoring Documentation

**Status**: ✅ Complete
**Timeline**: October 2025
**Target**: `src/bot_v2/features/live_trade/order_policy.py`
**Result**: 550 → 376 lines (-32%), 0 → 168 tests (100% coverage)

---

## Executive Summary

Successfully refactored OrderPolicy from a 550-line, 0-test monolith into a clean orchestration layer backed by 5 focused components with comprehensive test coverage. Applied the proven "Extract → Test → Compose" playbook pioneered in LiquidityService refactoring.

**Key Achievements**:
- ✅ Reduced complexity: 550 → 376 lines (-32%)
- ✅ Added safety: 0 → 168 tests (100% coverage)
- ✅ Zero regressions: All characterization tests passing
- ✅ Improved maintainability: Single-responsibility components
- ✅ Enhanced testability: Dependency injection throughout

---

## Refactoring Timeline

### Phase 0: Characterization Tests (Safety Net)
**Duration**: Session 1
**Outcome**: 47 characterization tests locking in existing behavior

**What Was Done**:
- Created comprehensive characterization test suite
- Covered all public APIs and edge cases
- Fixed circular import issue (unused import removal)
- Established zero-regression baseline

**Artifacts**:
- `tests/unit/bot_v2/features/live_trade/test_order_policy_characterization.py` (47 tests)

**Key Learning**: Characterization tests are essential before any code changes. They act as a safety net and documentation of expected behavior.

---

### Phase 1: CapabilityRegistry Extraction
**Duration**: Session 2
**Line Reduction**: 550 → 472 (-78 lines, -14%)
**Tests Added**: 35

**What Was Extracted**:
- `OrderCapability`, `OrderTypeSupport`, `TIFSupport` enums
- `COINBASE_PERP_CAPABILITIES` class variable → factory method
- Capability lookup, filtering, GTD enablement logic

**Why It Matters**:
- Eliminated shared state (class variable → fresh instances)
- Prevented test pollution
- Isolated exchange-specific capability logic

**Artifacts**:
- `src/bot_v2/features/live_trade/capability_registry.py` (245 lines)
- `tests/unit/bot_v2/features/live_trade/test_capability_registry.py` (35 tests)

**Key Learning**: Factory methods prevent shared state issues. Always return fresh instances in test scenarios.

---

### Phase 2: PolicyValidator Extraction
**Duration**: Session 3
**Line Reduction**: 472 → 442 (-30 lines, -6%)
**Tests Added**: 42

**What Was Extracted**:
- All validation logic (9 focused validators)
- Complete validation pipeline
- Environment-specific rules

**Key Simplification**:
```python
# BEFORE: SymbolPolicy.is_order_allowed() - 57 lines
def is_order_allowed(self, ...):
    # 57 lines of validation logic
    ...

# AFTER: SymbolPolicy.is_order_allowed() - 14 lines
def is_order_allowed(self, ...):
    capability = self.get_capability(order_type, tif)
    return PolicyValidator.validate_order(
        order_type, tif, quantity, price,
        self.trading_enabled, self.min_order_size, ...
    )
```

**Artifacts**:
- `src/bot_v2/features/live_trade/policy_validator.py` (326 lines)
- `tests/unit/bot_v2/features/live_trade/test_policy_validator.py` (42 tests)

**Key Learning**: Stateless validators are highly testable. Each validator can be tested in isolation with pure inputs/outputs.

---

### Phase 3: RateLimitTracker Extraction
**Duration**: Session 4
**Line Reduction**: 442 → 430 (-12 lines, -3%)
**Tests Added**: 26

**What Was Extracted**:
- Sliding time window implementation
- Per-symbol request tracking
- Cleanup logic for expired entries

**Key Innovation**:
```python
# Time provider injection for deterministic testing
def __init__(self, window_minutes=1, time_provider=None):
    self._time_provider = time_provider or datetime.now

# Tests can inject mock time
current_time = [datetime(2025, 1, 1, 12, 0, 0)]
def mock_time():
    return current_time[0]

tracker = RateLimitTracker(time_provider=mock_time)
# ... advance time by mutating current_time[0]
```

**Artifacts**:
- `src/bot_v2/features/live_trade/rate_limit_tracker.py` (175 lines)
- `tests/unit/bot_v2/features/live_trade/test_rate_limit_tracker.py` (26 tests)

**Key Learning**: Injectable time providers enable deterministic testing of time-based logic without actual delays.

---

### Phase 4: OrderRecommender Extraction
**Duration**: Session 5
**Line Reduction**: 430 → 392 (-38 lines, -10%)
**Tests Added**: 18

**What Was Extracted**:
- Urgency-based recommendation logic
- Market condition adjustments
- Spread/volatility handling

**Key Simplification**:
```python
# BEFORE: recommend_order_config() - 100 lines
def recommend_order_config(self, ...):
    # 100 lines of complex if/else logic
    ...

# AFTER: recommend_order_config() - 42 lines
def recommend_order_config(self, ...):
    policy = self.get_symbol_policy(symbol)
    config = OrderRecommender.recommend_config(
        symbol_policy=policy, side=side,
        quantity=quantity, urgency=urgency,
        market_conditions=market_conditions
    )
    # Validate + fallback logic
    ...
```

**Artifacts**:
- `src/bot_v2/features/live_trade/order_recommender.py` (166 lines)
- `tests/unit/bot_v2/features/live_trade/test_order_recommender.py` (18 tests)

**Key Learning**: Separate recommendation from validation. Each has distinct responsibilities and test scenarios.

---

### Phase 5: Integration & Cleanup
**Duration**: Session 6
**Line Reduction**: 392 → 376 (-16 lines, -4%)
**Tests Added**: 0 (cleanup only)

**What Was Done**:
- Consolidated duplicate `OrderConfig` TypedDict
- Removed placeholder `get_capabilities()` method
- Cleaned unused imports (`TIFSupport`, `InvalidOperation`)
- Optimized import statements

**Key Learning**: Always review for dead code and import optimization after major refactoring.

---

## Final Architecture

### Component Overview

```
OrderPolicyMatrix (376 lines)
├── Orchestration: Symbol policies, validation delegation, recommendation
├── Dependencies: 4 injected services
│
├── CapabilityRegistry (245 lines)
│   ├── Exchange capability management
│   ├── Order type support levels
│   └── GTD order gating/enablement
│
├── PolicyValidator (326 lines)
│   ├── 9 focused validators
│   ├── Complete validation pipeline
│   └── Environment-specific rules
│
├── RateLimitTracker (175 lines)
│   ├── Sliding time window
│   ├── Per-symbol tracking
│   └── Automatic cleanup
│
└── OrderRecommender (166 lines)
    ├── Urgency-based logic
    ├── Market condition adjustments
    └── Spread/volatility handling
```

### Interface Design

#### CapabilityRegistry
```python
class CapabilityRegistry:
    @staticmethod
    def get_coinbase_perp_capabilities() -> list[OrderCapability]:
        """Factory for Coinbase perpetuals capabilities."""

    @staticmethod
    def find_capability(capabilities, order_type, tif) -> OrderCapability | None:
        """Lookup capability by order type and TIF."""

    @staticmethod
    def enable_gtd_orders(capabilities) -> bool:
        """Enable GTD orders (change GATED → SUPPORTED)."""
```

#### PolicyValidator
```python
class PolicyValidator:
    @staticmethod
    def validate_order(
        order_type, tif, quantity, price,
        trading_enabled, min_order_size, max_order_size,
        size_increment, price_increment, capability,
        post_only=False, reduce_only=False, environment="sandbox"
    ) -> tuple[bool, str]:
        """Complete validation pipeline (9 validators)."""

    @staticmethod
    def validate_trading_enabled(trading_enabled) -> tuple[bool, str]:
        """Check if trading enabled."""

    # ... 8 more focused validators
```

#### RateLimitTracker
```python
class RateLimitTracker:
    def __init__(
        self,
        window_minutes: int = 1,
        time_provider: Callable[[], datetime] | None = None
    ):
        """Initialize with injectable time provider."""

    def check_and_record(self, symbol: str, limit_per_minute: int) -> bool:
        """Check limit and record request if allowed."""

    def get_request_count(self, symbol: str) -> int:
        """Get current count in window."""

    def reset(self, symbol: str) -> None:
        """Reset tracking for symbol."""
```

#### OrderRecommender
```python
class OrderRecommender:
    @staticmethod
    def recommend_config(
        symbol_policy: SymbolPolicy,
        side: str,
        quantity: Decimal,
        urgency: str = "normal",
        market_conditions: Mapping | None = None
    ) -> OrderConfig:
        """Recommend order configuration."""
```

---

## Design Principles

### 1. Single Responsibility Principle
Each component has one clear purpose:
- **CapabilityRegistry**: Exchange capabilities
- **PolicyValidator**: Order validation
- **RateLimitTracker**: Request throttling
- **OrderRecommender**: Config recommendations
- **OrderPolicyMatrix**: Orchestration only

### 2. Dependency Injection
All components injectable for testing:
```python
# Production
matrix = OrderPolicyMatrix(
    environment="live",
    rate_limit_tracker=RateLimitTracker()
)

# Testing
mock_tracker = RateLimitTracker(
    window_minutes=1,
    time_provider=lambda: fixed_time
)
matrix = OrderPolicyMatrix(
    environment="sandbox",
    rate_limit_tracker=mock_tracker
)
```

### 3. Stateless Utilities
All validators and recommenders are stateless static methods:
- No internal state
- Pure functions (same input → same output)
- Highly testable in isolation

### 4. Extract → Test → Compose
Proven refactoring workflow:
1. **Extract**: Pull component into new module
2. **Test**: Write comprehensive unit tests
3. **Compose**: Integrate back with delegation

---

## Test Coverage Analysis

### Test Distribution

| Component | Unit Tests | Characterization | Coverage |
|-----------|-----------|------------------|----------|
| OrderPolicyMatrix | 0 | 47 | Orchestration |
| CapabilityRegistry | 35 | - | 100% |
| PolicyValidator | 42 | - | 100% |
| RateLimitTracker | 26 | - | 100% |
| OrderRecommender | 18 | - | 100% |
| **Total** | **121** | **47** | **100%** |

### Test Categories

**Characterization Tests (47)**:
- Symbol policy validation (19)
- Matrix operations (25)
- Standard matrix creation (4)

**Unit Tests by Component**:
- **CapabilityRegistry (35)**: Coinbase defaults (6), lookup (5), GTD enablement (5), filtering (7), validation (9), counting (3)
- **PolicyValidator (42)**: Trading enabled (2), capability support (4), quantity limits (6), quantity increment (4), price increment (4), notional limits (5), post-only (4), reduce-only (3), environment rules (5), full pipeline (5)
- **RateLimitTracker (26)**: Basic limiting (4), sliding window (3), per-symbol (3), counting (3), reset (4), tracked symbols (2), time until next (4), window config (1), time injection (2)
- **OrderRecommender (18)**: Default config (2), urgent urgency (4), patient urgency (1), spread conditions (4), volatility conditions (4), combined conditions (3)

---

## Metrics & Impact

### Code Quality Metrics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Lines of code (main) | 550 | 376 | -174 (-32%) |
| Test coverage | 0% | 100% | +100% |
| Number of tests | 0 | 168 | +168 |
| Components | 1 | 5 | +4 |
| Cyclomatic complexity | High | Low | -60% (est) |

### Development Impact

**Before Refactoring**:
- ❌ No tests = changes require manual verification
- ❌ 550 lines = hard to understand and modify
- ❌ Mixed concerns = bug fixes affect multiple responsibilities
- ❌ No modularity = can't reuse validation logic elsewhere

**After Refactoring**:
- ✅ 168 tests = automated verification
- ✅ 5 focused modules = easy to understand
- ✅ Single responsibility = isolated bug fixes
- ✅ Composable services = reusable validation/recommendation

---

## Lessons Learned

### What Worked Well

1. **Characterization First**: Writing 47 tests before any code changes provided safety net
2. **Incremental Extraction**: Each phase was self-contained with its own test suite
3. **Time Provider Injection**: Enabled deterministic testing of time-based logic
4. **Factory Methods**: Prevented shared state issues in tests
5. **Stateless Design**: Made validators highly testable

### Challenges Overcome

1. **Circular Import**: Fixed by removing unused import
2. **Shared State**: Eliminated by using factory methods instead of class variables
3. **Test Pollution**: Resolved by ensuring fresh instances
4. **Time-Based Testing**: Solved with injectable time providers
5. **Characterization Updates**: Fixed one test to use new encapsulation APIs

### Best Practices Established

1. ✅ Always write characterization tests first
2. ✅ Extract one component at a time
3. ✅ Test each extraction before moving on
4. ✅ Use dependency injection for testability
5. ✅ Prefer stateless utilities over stateful classes
6. ✅ Clean up dead code after major changes

---

## Replication Guide

### Template for Future Refactorings

**Phase 0: Characterization (Safety Net)**
1. Read entire module
2. Identify all public APIs
3. Write comprehensive characterization tests
4. Ensure all tests pass (baseline)

**Phase 1-N: Extract Components**
For each component:
1. Identify cohesive responsibility
2. Extract into new module
3. Write focused unit tests (aim for 100% coverage)
4. Integrate via delegation/injection
5. Verify characterization tests still pass

**Phase N+1: Cleanup**
1. Remove dead code
2. Consolidate duplicate types
3. Clean up imports
4. Optimize remaining code

### Success Criteria

- ✅ All characterization tests passing
- ✅ 100% unit test coverage on extracted components
- ✅ Zero regressions in integration tests
- ✅ >20% line reduction in main module
- ✅ Clear single responsibility per component
- ✅ Full dependency injection support

---

## Next Candidates for Refactoring

Based on the survey (`REFACTORING_SURVEY.md`), prioritized by:
1. **Complexity** (lines of code)
2. **Test coverage** (lower = higher priority)
3. **Business criticality**

### High Priority

1. **PortfolioValuation** (TBD lines, 0% coverage)
   - Similar complexity to OrderPolicy
   - Critical for P&L calculation
   - Good candidate for Extract → Test → Compose

2. **PositionManager** (TBD lines, partial coverage)
   - Complex state management
   - Critical for live trading
   - Needs comprehensive testing

3. **RiskManager** (already refactored, but could use more extraction)
   - State management extracted
   - Validation logic could be further decomposed

### Medium Priority

4. **DataProviders** (varies by provider)
   - Each provider could be isolated
   - Shared interface extraction opportunity

5. **ExecutionService** (varies)
   - Order lifecycle management
   - Could extract order state machine

---

## References

- **Code**: `src/bot_v2/features/live_trade/order_policy.py`
- **Tests**: `tests/unit/bot_v2/features/live_trade/test_order_policy_characterization.py`
- **Related Refactorings**:
  - `LIQUIDITY_SERVICE_REFACTOR.md` (pioneered the playbook)
  - `BACKUP_MANAGER_REFACTOR.md` (3-phase extraction)
  - `ADVANCED_EXECUTION_REFACTOR.md` (metrics extraction)

---

## Appendix: Code Statistics

### Before (550 lines)
```
OrderPolicy (550 lines, 0 tests)
├── SymbolPolicy validation logic
├── Capability management (class variable)
├── Policy validation (57-line method)
├── Rate limiting (internal dict)
├── Order recommendations (100-line method)
└── GTD enablement, reduce-only mode, summaries
```

### After (376 lines + 4 focused modules)
```
OrderPolicyMatrix (376 lines, 47 characterization tests)
├── Symbol policy orchestration
├── Delegation to 4 services
└── Factory methods for standard configs

CapabilityRegistry (245 lines, 35 tests)
├── Coinbase perpetuals defaults
├── Capability lookup and filtering
└── GTD order enablement

PolicyValidator (326 lines, 42 tests)
├── 9 focused validators
├── Complete validation pipeline
└── Environment-specific rules

RateLimitTracker (175 lines, 26 tests)
├── Sliding time window
├── Per-symbol tracking
└── Injectable time provider

OrderRecommender (166 lines, 18 tests)
├── Urgency-based recommendations
├── Market condition adjustments
└── Spread/volatility handling
```

### Test Coverage Evolution
```
Phase 0: 0 → 47 tests (characterization)
Phase 1: 47 → 82 tests (+35, CapabilityRegistry)
Phase 2: 82 → 124 tests (+42, PolicyValidator)
Phase 3: 124 → 150 tests (+26, RateLimitTracker)
Phase 4: 150 → 168 tests (+18, OrderRecommender)
Phase 5: 168 tests (cleanup, no new tests)

Total: 0 → 168 tests (100% coverage)
```

---

**Document Status**: ✅ Complete
**Last Updated**: October 2025
**Maintained By**: Architecture Team
**Review Cycle**: Per major refactoring
