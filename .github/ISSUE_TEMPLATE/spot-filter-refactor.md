---
name: Refactor spot filter pipeline into composable filter strategies
about: Extract filter logic into testable, composable strategies
labels: refactoring, code-quality, strategy, enhancement
---

## Objective
Refactor `_apply_spot_filters()` into composable filter strategies to eliminate repeated window/multiplier boilerplate and make future filter rules easier to add.

## Current State
**Location:** `src/gpt_trader/features/live_trade/orchestrator/spot_filters.py:20-24`

Current implementation has:
- A no-op `SpotFiltersMixin._apply_spot_filters()` that returns the decision unchanged
- No filter registry or per-filter functions
- No dedicated unit tests for spot filter behavior

`SpotFiltersMixin` is invoked from `src/gpt_trader/features/live_trade/orchestrator/decision.py`.

## Proposed Design

### Filter Registry Pattern
```python
FilterFn = Callable[[dict, list[Decimal], ...], Decision | None]

FILTERS: dict[str, FilterFn] = {
    "volume_filter": _apply_volume_filter,
    "momentum_filter": _apply_momentum_filter,
    "trend_filter": _apply_trend_filter,
    "volatility_filter": _apply_volatility_filter,
}
```

### Filter Implementation
Each filter:
1. Extracts its own config from rules dict
2. Validates window size and data availability
3. Returns `Decision(action=Action.HOLD, reason=...)` to block entry, or `None` to continue
4. Is independently unit-testable with synthetic data

### Simplified Orchestration
```python
async def _apply_spot_filters(self, context, decision):
    # ... fetch candles ...
    rules = self._spot_profiles.get(context.symbol)

    for filter_name, filter_fn in FILTERS.items():
        if result := filter_fn(rules.get(filter_name), closes, volumes, highs, lows):
            return result  # Filter blocked entry

    return decision  # All filters passed
```

## Acceptance Criteria

### Functional Requirements
- [ ] Each filter extracted into separate function (`_apply_volume_filter`, `_apply_momentum_filter`, `_apply_trend_filter`, `_apply_volatility_filter`)
- [ ] Volume filter blocks when `latest_volume < avg_volume * multiplier`
- [ ] Momentum filter blocks when `RSI > oversold_threshold`
- [ ] Trend filter blocks when `slope < min_slope`
- [ ] Volatility filter blocks when `volatility_pct` outside `[min_vol, max_vol]` range
- [ ] All filters return `None` when config is missing/disabled
- [ ] All filters return appropriate `Decision(HOLD, reason=...)` when blocking

### Testing Requirements
- [ ] Create `tests/unit/gpt_trader/features/live_trade/orchestrator/test_spot_filters.py`
- [ ] Test each filter in isolation with synthetic candle data
- [ ] Verify volume filter with various multiplier values
- [ ] Verify momentum filter with RSI edge cases (exactly at threshold, just above/below)
- [ ] Verify trend filter with positive/negative/zero slopes
- [ ] Verify volatility filter with edge cases (min_vol, max_vol boundaries)
- [ ] Test empty/missing config handling for each filter
- [ ] Test insufficient data scenarios (window size > available candles)

### Quality Gates
- [ ] All existing integration tests pass (no behavior change)
- [ ] Code coverage maintained or improved
- [ ] Ruff/mypy/black checks pass
- [ ] No performance regression (filter execution time comparable to current implementation)

## Implementation Plan

### Phase 1: Extract Filter Functions
1. Create `_apply_volume_filter()` inside `spot_filters.py`
2. Create `_apply_momentum_filter()` inside `spot_filters.py`
3. Create `_apply_trend_filter()` inside `spot_filters.py`
4. Create `_apply_volatility_filter()` inside `spot_filters.py`

### Phase 2: Add Unit Tests
1. Create test file with synthetic candle data fixtures
2. Write tests for each filter (3-5 test cases per filter)
3. Ensure 100% coverage of filter logic

### Phase 3: Refactor Orchestrator
1. Define `FILTERS` registry dict
2. Replace inline logic with registry iteration
3. Verify all tests still pass

### Phase 4: Documentation
1. Add docstrings to each filter function
2. Update strategy orchestrator docstring
3. Add example usage in comments

## Files to Change

**Modified:**
- `src/gpt_trader/features/live_trade/orchestrator/spot_filters.py` - add registry and filter helpers

**Created:**
- `tests/unit/gpt_trader/features/live_trade/orchestrator/test_spot_filters.py` - comprehensive filter tests

## Related Work

This cleanup was identified during the broader code quality improvements tracked in:
- UTC datetime helpers refactor
- ExecutionCoordinator cleanup
- Centralized metric emission

## References

**Current implementation:** `src/gpt_trader/features/live_trade/orchestrator/spot_filters.py:20-24`

**Filter logic locations (post-implementation):**
- Volume filter: `spot_filters.py` helper
- Momentum filter: `spot_filters.py` helper
- Trend filter: `spot_filters.py` helper
- Volatility filter: `spot_filters.py` helper
