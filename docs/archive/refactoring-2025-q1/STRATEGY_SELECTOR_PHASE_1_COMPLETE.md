# Strategy Selector Refactor – Phase 1 Complete

**Date:** October 2025
**Phase:** SymbolUniverseBuilder Extraction
**Status:** ✅ Complete
**Pattern:** Extract → Test → Integrate → Validate

## Summary

Extracted symbol universe selection logic into `SymbolUniverseBuilder` with injectable universe source, improving testability and isolating tier-based filtering logic. Updated existing tests and verified zero regressions.

### Metrics
- `strategy_selector.py`: 544 → 505 lines (39 line reduction, 7.2%)
- `symbol_universe_builder.py`: 75 lines (new service)
- Tests: +8 SymbolUniverseBuilder tests, 2 existing tests updated
- Total adaptive portfolio tests: 119 → 127 (all green)
- **Zero regressions** - all 11 existing strategy selector tests passing

### Highlights
- `SymbolUniverseBuilder` with injectable universe source (default: 25 hardcoded symbols)
- `build_universe(tier_config, portfolio_snapshot)` method for tier-appropriate filtering
- Tier-based universe sizing: Micro (8), Small (12), Medium (18), Large (25)
- Removed 45-line `_get_symbol_universe` method from StrategySelector
- 8 focused unit tests for universe builder
- StrategySelector now uses injected builder via `self.universe_builder`

## Changes Made

### New Files

#### 1. `symbol_universe_builder.py` (75 lines)

**Purpose:** Builds tier-appropriate symbol universes for strategy selection

**Components:**
- `_default_universe_source()` - returns default 25-symbol list
- `SymbolUniverseBuilder` class with injectable universe source
- `build_universe()` method for tier-based filtering

**Design:**
```python
def _default_universe_source() -> list[str]:
    """Default symbol universe - simplified for production."""
    return ["AAPL", "MSFT", "GOOGL", ...]  # 25 symbols

class SymbolUniverseBuilder:
    def __init__(self, universe_source: Callable[[], list[str]] | None = None):
        if universe_source is None:
            universe_source = _default_universe_source
        self._universe_source = universe_source

    def build_universe(self, tier_config, portfolio_snapshot) -> list[str]:
        base_universe = self._universe_source()

        if tier_config.name == "Micro Portfolio":
            return base_universe[:8]
        elif tier_config.name == "Small Portfolio":
            return base_universe[:12]
        elif tier_config.name == "Medium Portfolio":
            return base_universe[:18]
        else:  # Large Portfolio or custom tiers
            return base_universe
```

**Benefits:**
- **Injectable source**: Can provide custom universe list for testing/production
- **Testable**: No hardcoded dependencies, pure filtering logic
- **Focused**: Single responsibility (universe selection)
- **Swappable**: Can use different universe providers (API, database, etc.)

#### 2. `test_symbol_universe_builder.py` (8 tests)

**Test Coverage:**

| Test Suite | Tests | Purpose |
|------------|-------|---------|
| Tier Filtering | 4 | Micro (8), Small (12), Medium (18), Large (25) |
| Custom Source | 1 | Injectable universe source works |
| Unknown Tier | 1 | Defaults to full universe |
| Default Source | 1 | Uses built-in 25 symbols when no source provided |
| Subset Verification | 1 | Smaller tiers return ordered subset |

**Test Categories:**
- ✅ test_micro_tier_gets_eight_symbols
- ✅ test_small_tier_gets_twelve_symbols
- ✅ test_medium_tier_gets_eighteen_symbols
- ✅ test_large_tier_gets_all_symbols
- ✅ test_uses_custom_universe_source
- ✅ test_unknown_tier_defaults_to_full_universe
- ✅ test_default_source_when_none_provided
- ✅ test_returns_subset_of_source_universe

### Modified Files

#### `strategy_selector.py` (544 → 505 lines, -39 lines)

**Before:**
- Hardcoded 25-symbol base universe in `_get_symbol_universe`
- Tier filtering logic embedded in method (45 lines)
- No injectable universe source

**After:**
- Import SymbolUniverseBuilder
- Injectable `universe_builder` parameter in `__init__` (default: new instance)
- Replaced `self._get_symbol_universe(tier_config, portfolio_snapshot)` with `self.universe_builder.build_universe(tier_config, portfolio_snapshot)`
- Removed entire `_get_symbol_universe` method (45 lines)

**Key Changes:**
```python
# Before
class StrategySelector:
    def __init__(self, config, data_provider):
        self.config = config
        self.data_provider = data_provider
        self.logger = logging.getLogger(__name__)

    def generate_signals(self, tier_config, portfolio_snapshot, market_data=None):
        symbols = self._get_symbol_universe(tier_config, portfolio_snapshot)
        # ...

    def _get_symbol_universe(self, tier_config, portfolio_snapshot):
        base_universe = ["AAPL", "MSFT", ...]  # 25 symbols
        if tier_config.name == "Micro Portfolio":
            return base_universe[:8]
        # ... 45 lines total

# After
from bot_v2.features.adaptive_portfolio.symbol_universe_builder import SymbolUniverseBuilder

class StrategySelector:
    def __init__(self, config, data_provider, universe_builder=None):
        self.config = config
        self.data_provider = data_provider
        self.universe_builder = universe_builder or SymbolUniverseBuilder()
        self.logger = logging.getLogger(__name__)

    def generate_signals(self, tier_config, portfolio_snapshot, market_data=None):
        symbols = self.universe_builder.build_universe(tier_config, portfolio_snapshot)
        # ...

    # _get_symbol_universe method removed (45 lines deleted)
```

#### `test_strategy_selector.py` (2 tests updated)

**Changes:**
- Updated `test_micro_tier_gets_limited_universe` to use `strategy_selector.universe_builder.build_universe()`
- Updated `test_large_tier_gets_full_universe` to use `strategy_selector.universe_builder.build_universe()`
- Both tests still verify same behavior (universe sizing by tier)

**Example:**
```python
# Before
def test_micro_tier_gets_limited_universe(self, strategy_selector, portfolio_snapshot):
    tier_config = TierConfig(name="Micro Portfolio", ...)
    universe = strategy_selector._get_symbol_universe(tier_config, portfolio_snapshot)
    assert len(universe) <= 8

# After
def test_micro_tier_gets_limited_universe(self, strategy_selector, portfolio_snapshot):
    tier_config = TierConfig(name="Micro Portfolio", ...)
    universe = strategy_selector.universe_builder.build_universe(tier_config, portfolio_snapshot)
    assert len(universe) <= 8
```

## Validation

### Test Results

**SymbolUniverseBuilder Tests:**
```bash
$ pytest tests/.../test_symbol_universe_builder.py -v
============================= 8 passed in 0.03s ==============================
```

**StrategySelector Tests (After Update):**
```bash
$ pytest tests/.../test_strategy_selector.py -v
============================= 11 passed in 0.03s ==============================
```

**Full Adaptive Portfolio Test Suite:**
```bash
$ pytest tests/unit/bot_v2/features/adaptive_portfolio/ -v
============================= 127 passed in 0.16s ==============================
```

**Zero regressions** - All 119 existing tests + 8 new tests passing

### Behavioral Verification

✅ **Universe sizing unchanged** - Micro (8), Small (12), Medium (18), Large (25)
✅ **Signal generation unchanged** - Same symbols passed to strategies
✅ **Tier filtering unchanged** - Same tier name matching logic
✅ **Unknown tier handling unchanged** - Defaults to full universe
✅ **Integration unchanged** - StrategySelector uses builder transparently

## Design Decisions

### 1. Injectable Universe Source

**Decision:** Inject callable that returns list[str], not the list itself

**Rationale:**
- Allows dynamic universe generation (API calls, database queries)
- Can mock source for testing without rebuilding builder
- Follows same pattern as CLI refactoring (injectable dependencies)
- Default source is pure function (no side effects)

### 2. Keep Tier Filtering in Builder

**Decision:** Tier name matching logic stays in `build_universe` method

**Rationale:**
- Simple string matching doesn't need extraction
- Filtering logic is core to universe building responsibility
- May be replaced with more sophisticated logic in future
- Keeps builder focused on single task

### 3. Default Universe as Module-Level Function

**Decision:** `_default_universe_source()` is module-level function, not method

**Rationale:**
- Stateless - doesn't need instance context
- Can be imported and tested independently
- Can be used as default in other contexts
- Clear separation: data vs. behavior

### 4. Accept portfolio_snapshot Parameter (Currently Unused)

**Decision:** Keep `portfolio_snapshot` parameter even though not currently used

**Rationale:**
- Future enhancement: filter universe by existing positions
- Future enhancement: exclude symbols with large positions
- Future enhancement: sector-based universe expansion
- Maintains consistent signature with original method

## Lessons Learned

### What Worked Well ✅

1. **Clean extraction pattern** - Same Extract → Test → Integrate → Validate from CLI
2. **Injectable dependencies** - Builder accepts custom universe source
3. **Test-first approach** - 8 builder tests written before integration
4. **Zero regressions** - Updated 2 existing tests, all 11 still pass

### Challenges Overcome 🔧

1. **Test failures after integration** - 2 tests called old `_get_symbol_universe` method
   - **Solution**: Updated tests to use `universe_builder.build_universe()`
   - **Lesson**: Search for all method references before deleting

2. **Line count less than expected** - Reduced 39 lines vs. expected 45
   - **Context**: Added import + injectable parameter + instance variable
   - **Tradeoff**: Testability > line count
   - **Result**: Still 7.2% reduction with full testability

### Phase 1 Impact 🎯

**Testability Gains:**
- ✅ Can test universe selection without StrategySelector
- ✅ Can test tier filtering in isolation
- ✅ Can inject custom universe for testing
- ✅ No data provider mocking needed for universe tests

**Code Quality:**
- ✅ Universe selection isolated in dedicated component
- ✅ Single responsibility for builder
- ✅ Dependency injection pattern applied
- ✅ StrategySelector focuses on orchestration

## Next Steps

### Phase 2: PositionSizeCalculator (Planned)

**Goal:** Extract position sizing logic

**Scope:**
- Create `PositionSizeCalculator` class
- Extract `_calculate_signal_position_size` method (~21 lines)
- Injectable tier config for sizing rules
- ~6 focused tests

**Expected:**
- Remove ~21 lines from strategy_selector.py
- Add `position_size_calculator.py` (~40 lines including tests)
- Reusable sizing logic across strategies

### Remaining Phases

**Phase 3:** SignalFilter (~60 lines, +10 tests)
**Phase 4:** Strategy Implementations (~280 lines, +30 tests, **highest impact**)
**Phase 5:** StrategySignalRouter (~22 lines, +6 tests)
**Phase 6:** SignalRanker (~8 lines, +3 tests, optional)

### Final State (After All Phases)

**Projected:**
- strategy_selector.py: ~150-185 lines (orchestration only)
- Total new tests: ~63 (8 universe + 6 sizing + 10 filter + 30 strategies + 6 router + 3 ranker)
- Total reduction: ~65-70% from baseline (544 → ~180 lines)
- Full testability for all components

## Pattern Evolution

### CLI Refactoring → Strategy Selector

**Shared Patterns:**
- ✅ Extract → Test → Integrate → Validate workflow
- ✅ Injectable dependencies (printer, runner, universe source)
- ✅ Default instances when not provided
- ✅ Focused unit tests for extracted components
- ✅ Zero regressions in existing tests
- ✅ Line count reduction secondary to testability

**Adaptations:**
- CLI: Sync commands with printer injection
- Strategy: Pure functions with callable injection
- CLI: Command handlers orchestrate services
- Strategy: StrategySelector orchestrates builders

---

**Phase 1 Status:** ✅ Complete
**Line Count:** 505 (39 line reduction from 544, 7.2%)
**Test Coverage:** ✅ 127 tests passing (8 new universe builder tests)
**Zero Regressions:** ✅ Confirmed
**Ready for Phase 2:** ✅ PositionSizeCalculator extraction
