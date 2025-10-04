# Strategy Selector Refactor – Phase 0 Baseline

**Date:** October 2025
**Target:** src/bot_v2/features/adaptive_portfolio/strategy_selector.py
**Pattern:** Extract → Test → Integrate → Validate
**Goal:** Modularize strategy selection, signal generation, filtering, and ranking

## Current State

### File Metrics
- **Lines:** 544 (target: ~150-200 lines, 63-72% reduction)
- **Test Coverage:** 11 tests (minimal - integration-style)
- **Responsibilities:** 7 major concerns (universe selection, strategy routing, 4 strategies, filtering, ranking, position sizing)
- **Dependencies:** DataProvider, PortfolioConfig, TierConfig, PortfolioSnapshot types

### Responsibilities Analysis

**1. Symbol Universe Selection** (lines 107-151, ~45 lines)
```python
def _get_symbol_universe(tier_config, portfolio_snapshot) -> list[str]
```
- Returns tier-appropriate symbol list
- Hardcoded base universe (25 symbols)
- Tier-based filtering: Micro (8), Small (12), Medium (18), Large (25)
- **Extract Candidate:** ✅ SymbolUniverseBuilder

**2. Strategy Signal Generation Router** (lines 152-173, ~22 lines)
```python
def _generate_strategy_signals(strategy_name, symbols, tier_config, portfolio_snapshot, market_data)
```
- Routes to appropriate strategy implementation
- 4 strategies: momentum, mean_reversion, trend_following, ml_enhanced
- Returns empty list for unknown strategies
- **Extract Candidate:** ✅ StrategySignalRouter

**3. Individual Strategy Implementations** (~280 lines total)

**Momentum Strategy** (lines 174-227, ~54 lines)
- 5-day and 20-day returns calculation
- Buy signal: returns_5d > 2% AND returns_20d > 5%
- Confidence calculation based on returns
- Position sizing via _calculate_signal_position_size

**Mean Reversion Strategy** (lines 228-291, ~64 lines)
- 20-day SMA and standard deviation calculation
- Z-score calculation
- Buy signal: z_score < -1.5 (oversold)
- Confidence based on z-score magnitude

**Trend Following Strategy** (lines 292-357, ~66 lines)
- 10/30/50-day moving averages
- Buy signal: SMA alignment (10 > 30 > 50) + price > SMA10
- Trend strength calculation
- Confidence based on trend strength

**ML Enhanced Strategy** (lines 359-387, ~29 lines)
- Wraps momentum strategy
- Filters high-confidence signals (>0.6)
- Boosts confidence by 20% (capped at 0.95)
- Simplified ML simulation

**Extract Candidate:** ✅ Individual strategy classes (MomentumStrategy, MeanReversionStrategy, etc.)

**4. Signal Filtering** (lines 389-423, ~35 lines)
```python
def _filter_signals(signals, tier_config, portfolio_snapshot) -> list[TradingSignal]
```
- Filters existing positions
- Minimum confidence threshold per tier
- Minimum position size check
- Market constraints check (excluded symbols)
- **Extract Candidate:** ✅ SignalFilter

**5. Signal Ranking** (lines 425-432, ~8 lines)
```python
def _rank_signals(signals, tier_config) -> list[TradingSignal]
```
- Simple confidence-based ranking
- Sorts descending by confidence
- **Extract Candidate:** ✅ SignalRanker (or keep inline if trivial)

**6. Position Sizing Calculator** (lines 450-470, ~21 lines)
```python
def _calculate_signal_position_size(confidence, tier_config, portfolio_snapshot) -> float
```
- Base size: portfolio_value / target_positions
- Confidence adjustment
- Minimum size enforcement (tier.min_position_size)
- Maximum size cap (25% of portfolio)
- **Extract Candidate:** ✅ PositionSizeCalculator

**7. Helper Methods** (~60 lines)
- `_calculate_max_signals` (lines 434-448, ~15 lines) - max new signals for tier
- `_get_min_confidence_for_tier` (lines 472-485, ~14 lines) - tier-based confidence threshold
- `_meets_market_constraints` (lines 487-499, ~13 lines) - excluded symbol check
- `_safe_get_price` (lines 42-61, ~20 lines) - pandas/SimpleDataFrame compatibility
- `get_strategy_allocation` (lines 501-543, ~43 lines) - strategy allocation percentages

### Current Test Coverage (11 tests)

**Initialization (1 test)**
- ✅ test_initializes_with_config_and_provider

**Signal Generation (2 tests)**
- ✅ test_generates_signals_for_tier
- ✅ test_respects_tier_strategy_list

**Momentum Strategy (1 test)**
- ✅ test_generates_momentum_signals

**Symbol Universe (2 tests)**
- ✅ test_micro_tier_gets_limited_universe
- ✅ test_large_tier_gets_full_universe

**Position Sizing (2 tests)**
- ✅ test_calculates_position_size_based_on_confidence
- ✅ test_respects_minimum_position_size

**Signal Filtering (1 test)**
- ✅ test_limits_signals_to_tier_capacity

**Error Handling (2 tests)**
- ✅ test_handles_insufficient_data_gracefully
- ✅ test_handles_data_provider_exceptions

**Coverage Gaps:**
- ❌ No tests for mean_reversion strategy
- ❌ No tests for trend_following strategy
- ❌ No tests for ml_enhanced strategy
- ❌ No tests for signal ranking
- ❌ No tests for signal filtering logic (only capacity test)
- ❌ No tests for _calculate_max_signals
- ❌ No tests for _get_min_confidence_for_tier
- ❌ No tests for _meets_market_constraints
- ❌ No tests for get_strategy_allocation
- ❌ No tests for _safe_get_price

### Dependencies

**External:**
- `DataProvider` (get_historical_data method)
- `pandas` (optional - can work with SimpleDataFrame)
- `logging`

**Internal Types:**
- `PortfolioConfig` - overall portfolio configuration
- `TierConfig` - tier-specific configuration
- `PortfolioSnapshot` - current portfolio state
- `TradingSignal` - signal output dataclass

### Current Design Issues

**1. Tight Coupling**
- All strategies embedded in single class
- No strategy interface/protocol
- Hard to test strategies in isolation
- Hard to add new strategies without modifying StrategySelector

**2. Mixed Responsibilities**
- Signal generation + filtering + ranking + position sizing all in one class
- Universe selection coupled with strategy execution
- Violates Single Responsibility Principle

**3. Limited Testability**
- Integration-style tests (full flow)
- Cannot test strategies independently
- Cannot test filtering/ranking in isolation
- Mock data provider required for all tests

**4. Hardcoded Logic**
- Base universe hardcoded in _get_symbol_universe
- Tier thresholds hardcoded in _get_min_confidence_for_tier
- Strategy allocation logic embedded in get_strategy_allocation

**5. Duplication**
- All 4 strategies have similar structure:
  - Get historical data
  - Calculate indicators
  - Check conditions
  - Create signal with position sizing
- pandas/SimpleDataFrame compatibility duplicated in each strategy

## Extraction Plan

### Phase 1: SymbolUniverseBuilder (Lowest Risk)
**Target:** Extract _get_symbol_universe into SymbolUniverseBuilder
**Lines:** ~45 lines → new service (~60 lines including tests)
**Risk:** Low - pure function, no external dependencies
**Benefits:**
- Isolate universe selection logic
- Easy to test symbol selection rules
- Can inject different universe providers later
- Clear separation of concerns

**Scope:**
- Create `SymbolUniverseBuilder` class
- Move hardcoded base universe to builder
- Injectable universe source (default: hardcoded list)
- Tier-based filtering logic
- ~8 focused tests

**Expected Reduction:** strategy_selector.py: 544 → ~510 lines

### Phase 2: PositionSizeCalculator
**Target:** Extract position sizing logic
**Lines:** ~21 lines → new service (~40 lines including tests)
**Risk:** Low - pure calculation, well-defined inputs/outputs
**Benefits:**
- Isolate position sizing logic
- Easy to test sizing rules
- Reusable across strategies
- Clear financial calculations

**Scope:**
- Create `PositionSizeCalculator` class
- Move _calculate_signal_position_size
- Injectable configuration (tier config)
- ~6 focused tests

**Expected Reduction:** strategy_selector.py: ~510 → ~495 lines

### Phase 3: SignalFilter
**Target:** Extract signal filtering logic
**Lines:** ~35 lines + helpers (~60 total) → new service (~90 lines including tests)
**Risk:** Medium - depends on portfolio state
**Benefits:**
- Isolate filtering rules
- Test filtering logic independently
- Easier to add new filter criteria
- Clear signal validation

**Scope:**
- Create `SignalFilter` class
- Move _filter_signals, _get_min_confidence_for_tier, _meets_market_constraints
- Injectable market constraints
- ~10 focused tests (filter rules, confidence thresholds, constraints)

**Expected Reduction:** strategy_selector.py: ~495 → ~440 lines

### Phase 4: Strategy Implementations (Highest Impact)
**Target:** Extract 4 strategy implementations into separate classes
**Lines:** ~280 lines → 4 strategy files (~100 lines each including tests)
**Risk:** Medium-High - core business logic
**Benefits:**
- Each strategy testable in isolation
- Easy to add new strategies
- Strategy interface/protocol
- Reduced StrategySelector complexity

**Scope:**
- Create `BaseStrategy` protocol/abstract class
- Create `MomentumStrategy` class (~70 lines + 8 tests)
- Create `MeanReversionStrategy` class (~80 lines + 8 tests)
- Create `TrendFollowingStrategy` class (~80 lines + 8 tests)
- Create `MLEnhancedStrategy` class (~50 lines + 6 tests)
- Move _safe_get_price to shared utility or base class
- ~30 total strategy tests

**Expected Reduction:** strategy_selector.py: ~440 → ~200 lines

### Phase 5: StrategySignalRouter
**Target:** Extract strategy routing/dispatch logic
**Lines:** ~22 lines → new service (~40 lines including tests)
**Risk:** Low - simple dispatch logic
**Benefits:**
- Strategy registry pattern
- Dynamic strategy loading
- Easy to add new strategies without modifying router

**Scope:**
- Create `StrategySignalRouter` class
- Strategy registry (dict[str, BaseStrategy])
- Injectable strategy instances
- ~6 focused tests

**Expected Reduction:** strategy_selector.py: ~200 → ~185 lines

### Phase 6: SignalRanker (Optional Polish)
**Target:** Extract signal ranking logic
**Lines:** ~8 lines → inline or small service
**Risk:** Low - trivial logic
**Decision:** Keep inline or extract if ranking becomes complex

**Expected Final State:** strategy_selector.py: ~150-185 lines (65-72% reduction)

## Multi-Phase Summary

| Phase | Target | Lines Extracted | New Tests | Risk | Reduction |
|-------|--------|----------------|-----------|------|-----------|
| 0 | Baseline | - | 11 existing | - | 544 lines |
| 1 | SymbolUniverseBuilder | ~45 | +8 | Low | 544 → ~510 |
| 2 | PositionSizeCalculator | ~21 | +6 | Low | ~510 → ~495 |
| 3 | SignalFilter | ~60 | +10 | Medium | ~495 → ~440 |
| 4 | Strategy Implementations | ~280 | +30 | Medium-High | ~440 → ~200 |
| 5 | StrategySignalRouter | ~22 | +6 | Low | ~200 → ~185 |
| 6 | SignalRanker (optional) | ~8 | +3 | Low | ~185 → ~180 |
| **Total** | **6 components** | **~436 lines** | **+63 tests** | - | **67% reduction** |

## Success Criteria

### Metrics
- ✅ StrategySelector reduced to ~150-200 lines (orchestration only)
- ✅ 60+ new focused tests (total: 74+ tests)
- ✅ Zero regressions in existing 11 tests
- ✅ Each strategy testable in isolation
- ✅ All filtering/sizing/universe logic testable independently

### Design Goals
- ✅ Single Responsibility Principle for all components
- ✅ Strategy interface/protocol for extensibility
- ✅ Injectable dependencies throughout
- ✅ No hardcoded business logic in StrategySelector
- ✅ Clear separation: universe → signals → filter → rank → size

### Testability
- ✅ Can test strategies without data provider
- ✅ Can test filtering without running strategies
- ✅ Can test position sizing independently
- ✅ Can test universe selection in isolation
- ✅ Mock-free unit tests where possible

## Recommended Approach

**Start with Phase 1: SymbolUniverseBuilder**

**Why?**
1. **Lowest Risk:** Pure function, no complex dependencies
2. **Clear Boundaries:** Well-defined input (tier_config) → output (symbol list)
3. **Immediate Value:** Universe selection testable in isolation
4. **Confidence Builder:** Success proves pattern works for this codebase
5. **Non-Breaking:** Easy to integrate without touching strategy logic

**Next Steps:**
1. ✅ Document Phase 0 baseline (this document)
2. Create SymbolUniverseBuilder class with injectable universe source
3. Add 8 focused tests (micro/small/medium/large tiers, custom universe, etc.)
4. Update StrategySelector to use SymbolUniverseBuilder
5. Update existing integration tests to inject builder
6. Verify zero regressions (11 existing tests still pass)
7. Proceed to Phase 2

---

**Phase 0 Status:** ✅ Complete
**Current State:** 544 lines, 11 tests
**Target State:** ~150-200 lines, 74+ tests (67% reduction)
**Ready for Phase 1:** ✅ SymbolUniverseBuilder extraction
