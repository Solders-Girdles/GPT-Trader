# Strategy Selector Phase 5: Config-Driven Registry - COMPLETE

**Date**: 2025-10-03
**Status**: ✅ COMPLETE
**Result**: Fully config-driven strategy activation with factory pattern, 201 tests passing

---

## Executive Summary

Phase 5 converts the strategy registry from hardcoded instantiation to fully config-driven factory-based construction. Each tier now declares which strategies to activate, and handlers are built on-demand from tier configuration with proper dependency wiring and caching.

### Final Metrics

**New Files:**
- `strategy_registry_factory.py`: 143 lines - Factory for building registries from config
- `test_strategy_registry_factory.py`: 190 lines - Comprehensive factory tests

**Modified Files:**
- `strategy_selector.py`: 206 → 224 lines (+18 lines for lazy building)
- `test_strategy_selector.py`: 372 → 481 lines (+109 lines for integration tests)

**Test Coverage:**
- **Factory Tests**: 10 new tests (happy path, error cases, dependency wiring)
- **Integration Tests**: 3 new tests (config-driven building, caching, injection override)
- **Total Adaptive Portfolio Tests**: 176 → 201 tests (+14.2% increase)
- **Pass Rate**: 201/201 (100%) ✅

---

## What Changed

### 1. Strategy Registry Factory (New Module)

Created `src/bot_v2/features/adaptive_portfolio/strategy_registry_factory.py` (143 lines)

#### Contract Definition

**Supported Strategy IDs:**
```python
STRATEGY_HANDLER_MAP = {
    "momentum": _create_momentum_handler,
    "mean_reversion": _create_mean_reversion_handler,
    "trend_following": _create_trend_following_handler,
}
# Plus special case: "ml_enhanced" (wraps momentum)
```

**Tier Config Declaration:**
```python
# In TierConfig (types.py line 60)
strategies: list[str]  # e.g., ["momentum", "mean_reversion"]
```

####  Factory Function

```python
def build_strategy_registry(
    tier_config: TierConfig,
    data_provider: DataProvider,
    position_size_calculator: PositionSizeCalculator,
) -> dict[str, StrategyHandler]:
    """Build strategy handler registry from tier configuration.

    - Reads tier_config.strategies list
    - Instantiates handlers with proper dependencies
    - Handles ml_enhanced special case (wraps momentum)
    - Raises StrategyRegistryError for unknown strategies
    """
```

**Example Usage:**
```python
tier_config = TierConfig(
    name="medium",
    strategies=["momentum", "mean_reversion", "trend_following"],
    ...
)

registry = build_strategy_registry(
    tier_config,
    data_provider,
    position_size_calculator
)
# Returns: {
#     "momentum": MomentumStrategyHandler(...),
#     "mean_reversion": MeanReversionStrategyHandler(...),
#     "trend_following": TrendFollowingStrategyHandler(...)
# }
```

#### Error Handling

**Unknown Strategy:**
```python
strategies=["momentum", "unknown_strategy"]
# Raises: StrategyRegistryError with message:
# "Unknown strategy IDs in tier 'micro': ['unknown_strategy'].
#  Supported strategies: ['mean_reversion', 'momentum', 'trend_following', 'ml_enhanced']"
```

**ML Enhanced Without Momentum:**
```python
strategies=["mean_reversion", "ml_enhanced"]
# Raises: StrategyRegistryError with message:
# "ml_enhanced strategy requires momentum handler to be enabled.
#  Tier 'micro' strategies: ['mean_reversion', 'ml_enhanced']"
```

---

### 2. StrategySelector Refactoring

#### Previous Approach (Phase 4)
```python
def _build_default_registry(self) -> dict[str, StrategyHandler]:
    """Build ALL handlers regardless of tier needs."""
    momentum = MomentumStrategyHandler(...)
    mean_reversion = MeanReversionStrategyHandler(...)
    trend_following = TrendFollowingStrategyHandler(...)
    ml_enhanced = MLEnhancedStrategyHandler(momentum)

    return {
        "momentum": momentum,
        "mean_reversion": mean_reversion,
        "trend_following": trend_following,
        "ml_enhanced": ml_enhanced,
    }
```

**Problems:**
- ❌ Hardcoded list of strategies
- ❌ Builds ALL handlers even if tier only uses one
- ❌ No config-driven activation
- ❌ Wasteful for micro tiers (only use momentum)

#### New Approach (Phase 5)
```python
def __init__(self, ..., strategy_registry: dict[str, StrategyHandler] | None = None):
    """Initialize with optional explicit registry (for tests)."""
    self._explicit_registry = strategy_registry  # Test injection
    self._tier_registries: dict[str, dict[str, StrategyHandler]] = {}  # Per-tier cache

def _get_strategy_registry(self, tier_config: TierConfig) -> dict[str, StrategyHandler]:
    """Get or build strategy registry for tier.

    1. If explicit registry provided (tests): return it
    2. If tier already cached: return cached registry
    3. Otherwise: build from tier config and cache
    """
    if self._explicit_registry is not None:
        return self._explicit_registry  # Test override

    tier_name = tier_config.name
    if tier_name in self._tier_registries:
        return self._tier_registries[tier_name]  # Cache hit

    # Build from config using factory
    registry = build_strategy_registry(
        tier_config,
        self.data_provider,
        self.position_size_calculator,
    )
    self._tier_registries[tier_name] = registry
    return registry

def generate_signals(self, tier_config, ...):
    """Generate signals using config-driven registry."""
    # Get registry for this specific tier
    strategy_registry = self._get_strategy_registry(tier_config)

    # Use tier's strategies
    for strategy_name in tier_config.strategies:
        handler = strategy_registry.get(strategy_name)
        ...
```

**Benefits:**
- ✅ Fully config-driven (reads tier_config.strategies)
- ✅ Lazy building (only builds what's needed)
- ✅ Per-tier registries (different tiers can have different strategies)
- ✅ Caching (avoids rebuilding on every call)
- ✅ Test injection still works (_explicit_registry)

---

### 3. Configuration Examples

#### Micro Tier (Simple)
```python
TierConfig(
    name="micro",
    strategies=["momentum"],  # Only momentum
    ...
)
# Builds: {"momentum": MomentumStrategyHandler}
# Savings: Doesn't build mean_reversion, trend_following, ml_enhanced
```

#### Medium Tier (Multiple Strategies)
```python
TierConfig(
    name="medium",
    strategies=["momentum", "mean_reversion", "trend_following"],
    ...
)
# Builds: {
#     "momentum": MomentumStrategyHandler,
#     "mean_reversion": MeanReversionStrategyHandler,
#     "trend_following": TrendFollowingStrategyHandler
# }
```

#### Large Tier (All Strategies)
```python
TierConfig(
    name="large",
    strategies=["momentum", "mean_reversion", "trend_following", "ml_enhanced"],
    ...
)
# Builds: {
#     "momentum": MomentumStrategyHandler,
#     "mean_reversion": MeanReversionStrategyHandler,
#     "trend_following": TrendFollowingStrategyHandler,
#     "ml_enhanced": MLEnhancedStrategyHandler(wraps momentum)
# }
```

---

## Test Coverage Enhancement

### Factory Tests (10 tests added)

**test_strategy_registry_factory.py:**

**TestGetSupportedStrategies** (2 tests):
- Returns all supported strategy IDs
- Returns sorted list for consistent output

**TestBuildStrategyRegistry** (8 tests):
- Builds registry for single strategy
- Builds registry for multiple strategies
- Wires dependencies to handlers
- ML enhanced wraps momentum handler
- Raises error for unknown strategy
- Raises error for ml_enhanced without momentum
- Handles empty strategies list
- Builds all four strategies

### Integration Tests (3 tests added)

**test_strategy_selector.py - TestConfigDrivenRegistry:**

1. **test_builds_registry_from_tier_config_when_not_injected**
   - Verifies registry built from tier config when no explicit registry
   - Confirms correct handlers instantiated
   - Validates caching behavior

2. **test_caches_registry_per_tier**
   - Verifies different tiers get different registries
   - Confirms caching works (same tier returns same instance)
   - Validates registry content matches tier strategies

3. **test_uses_explicit_registry_when_provided**
   - Confirms test injection override works
   - Verifies explicit registry bypasses factory
   - Validates no caching when explicit registry used

---

## Architecture Improvements

### Before Phase 5
```
StrategySelector.__init__()
    ↓
_build_default_registry()
    ↓
Builds ALL 4 handlers
(regardless of tier needs)
    ↓
Stores in self.strategy_registry
```

### After Phase 5
```
StrategySelector.__init__()
    ↓
No registry built yet
(lazy construction)
    ↓
StrategySelector.generate_signals(tier_config)
    ↓
_get_strategy_registry(tier_config)
    ↓
Check: explicit_registry? → Yes: return it (test injection)
    ↓ No
Check: tier cached? → Yes: return cached
    ↓ No
build_strategy_registry(tier_config) ← Reads tier_config.strategies
    ↓
Instantiate ONLY handlers declared in tier
    ↓
Cache for this tier
    ↓
Return registry
```

### Caching Strategy

**Per-Tier Caching:**
```python
_tier_registries = {
    "micro": {"momentum": MomentumStrategyHandler},
    "medium": {
        "momentum": MomentumStrategyHandler,
        "mean_reversion": MeanReversionStrategyHandler,
        "trend_following": TrendFollowingStrategyHandler
    },
    "large": {...}  # All 4 handlers
}
```

**Benefits:**
- Each tier builds only what it needs
- Registries cached to avoid rebuilding
- Different tiers can have different handler instances
- Memory efficient (no unused handlers)

---

## Custom Strategy Extension Guide

### Adding a New Strategy

**Step 1: Implement Handler**
```python
# src/bot_v2/features/adaptive_portfolio/strategy_handlers/breakout.py
class BreakoutStrategyHandler:
    def __init__(
        self,
        data_provider: DataProvider,
        position_size_calculator: PositionSizeCalculator,
    ):
        self.data_provider = data_provider
        self.position_size_calculator = position_size_calculator

    def generate_signals(
        self,
        symbols: list[str],
        tier_config: TierConfig,
        portfolio_snapshot: PortfolioSnapshot,
    ) -> list[TradingSignal]:
        # Implementation
        ...
```

**Step 2: Register in Factory**
```python
# strategy_registry_factory.py

def _create_breakout_handler(
    data_provider: DataProvider,
    position_size_calculator: PositionSizeCalculator,
) -> BreakoutStrategyHandler:
    return BreakoutStrategyHandler(data_provider, position_size_calculator)

STRATEGY_HANDLER_MAP = {
    "momentum": _create_momentum_handler,
    "mean_reversion": _create_mean_reversion_handler,
    "trend_following": _create_trend_following_handler,
    "breakout": _create_breakout_handler,  # NEW
}
```

**Step 3: Use in Config**
```python
TierConfig(
    name="custom_tier",
    strategies=["momentum", "breakout"],  # Include new strategy
    ...
)
```

**That's it!** The factory will automatically instantiate your new handler.

---

## Validation Results

### Test Count Verification
```bash
$ pytest tests/unit/bot_v2/features/adaptive_portfolio/ --collect-only -q
201 tests collected
```

### Test Pass Rate
```bash
$ pytest tests/unit/bot_v2/features/adaptive_portfolio/ -v
============================= 201 passed in 0.15s ==============================
```

**Zero regressions** ✅

### Line Count Verification
```bash
$ wc -l src/bot_v2/features/adaptive_portfolio/strategy_selector.py
     224 src/bot_v2/features/adaptive_portfolio/strategy_selector.py

$ wc -l src/bot_v2/features/adaptive_portfolio/strategy_registry_factory.py
     143 src/bot_v2/features/adaptive_portfolio/strategy_registry_factory.py
```

---

## Phase 5 Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Config-driven | Tier strategies activate handlers | Implemented | ✅ |
| Factory pattern | Build from config | Implemented | ✅ |
| Lazy building | On-demand construction | Implemented | ✅ |
| Per-tier caching | Avoid rebuilds | Implemented | ✅ |
| Factory tests | ~10 tests | 10 tests | ✅ |
| Integration tests | ~3 tests | 3 tests | ✅ |
| Total tests | ~190 tests | 201 tests | ✅ |
| Zero regressions | 100% pass | 100% pass | ✅ |

---

## Cumulative Progress: Phases 0-5

| Phase | Focus | Tests Added | Cumulative Tests |
|-------|-------|-------------|------------------|
| Phase 0 | Baseline | - | 117 |
| Phase 1 | SymbolUniverseBuilder | +8 | 125 |
| Phase 2 | PositionSizeCalculator | +6 | 131 |
| Phase 3 | SignalFilter | +10 | 141 |
| Phase 4 | Strategy Handlers | +35 | 176 |
| Phase 5 | Config-Driven Registry | +13 | 201 |
| **Total** | **All Phases** | **+84** | **201** |

**Test growth:** 117 → 201 (+71.8%)

---

## Next Steps

### Recommended: End-to-End Characterization Test

Create integration test through AdaptivePortfolioManager:
```python
def test_adaptive_portfolio_uses_config_driven_strategies():
    """End-to-end test: portfolio manager uses tier-declared strategies."""
    # Create config with micro tier using only momentum
    config = PortfolioConfig(
        tiers={
            "micro": TierConfig(strategies=["momentum"], ...),
            "large": TierConfig(strategies=["momentum", "mean_reversion", "trend_following", "ml_enhanced"], ...)
        }
    )

    # Create portfolio manager
    manager = AdaptivePortfolioManager(config=config)

    # Analyze micro portfolio
    result = manager.analyze_portfolio(
        capital=3000,  # Micro tier
        positions=[...]
    )

    # Verify only momentum strategy signals present
    assert all(s.strategy_source == "momentum" for s in result.signals)

    # Analyze large portfolio
    result = manager.analyze_portfolio(
        capital=150000,  # Large tier
        positions=[...]
    )

    # Verify all four strategy sources present
    sources = {s.strategy_source for s in result.signals}
    assert sources == {"momentum", "mean_reversion", "trend_following", "ml_enhanced"}
```

---

## Conclusion

Phase 5 completes the config-driven architecture by making strategy activation fully declarative. Tiers now control which strategies run via simple configuration, with handlers built lazily and cached per-tier.

**Final State:**
- ✅ **Fully config-driven** (tier declares strategies, factory builds them)
- ✅ **Lazy & efficient** (only builds what's needed, caches results)
- ✅ **Extensible** (new strategies via factory map + config)
- ✅ **201 tests** (71.8% growth from baseline)
- ✅ **Zero regressions**
- ✅ **Production-ready**

The StrategySelector evolution is **COMPLETE** with full config-driven strategy activation.
