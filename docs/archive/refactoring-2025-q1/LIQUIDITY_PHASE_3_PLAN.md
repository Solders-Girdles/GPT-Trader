# LiquidityService Phase 3: ImpactEstimator Extraction Plan

**Status:** 📋 Ready for Review
**Target:** `estimate_market_impact()` method (~110 lines)
**Estimated Output:** ~140 lines (ImpactEstimator) + 15-18 tests

---

## Extraction Scope

### What Gets Extracted

**Core Logic** (~110 lines from liquidity_service.py:269-379):
1. **Square-root impact model** - Base impact calculation
2. **Depth adjustment** - Notional vs available depth multiplier
3. **Spread adjustment** - Spread-based impact inflation
4. **Condition multipliers** - EXCELLENT (0.5x) → CRITICAL (3.0x)
5. **Price calculations** - Estimated avg price, max impact price
6. **Slippage cost** - Dollar cost of market impact
7. **Execution recommendations** - Slicing threshold, post-only logic
8. **Conservative fallback** - Safe estimate when no depth data

### What Stays in LiquidityService

- `analyze_order_book()` - Already delegated to DepthAnalyzer
- Liquidity scoring (`_score_spread`, `_score_depth`, etc.) - Phase 4
- Latest analysis caching
- Snapshot generation

---

## Component Design

### ImpactEstimator Class

```python
class ImpactEstimator:
    """
    Estimates market impact for order execution.

    Uses square-root impact model with adjustments for:
    - Available market depth
    - Current spread conditions
    - Overall liquidity state
    """

    def __init__(self, max_impact_bps: Decimal = Decimal("50")):
        """Initialize with max acceptable impact threshold."""
        self.max_impact_bps = max_impact_bps

    def estimate(
        self,
        side: str,
        quantity: Decimal,
        analysis: DepthAnalysis,
        volume_metrics: dict[str, Decimal | int],
    ) -> ImpactEstimate:
        """Estimate market impact with execution recommendations."""
        # 1. Calculate base impact (square-root model)
        # 2. Apply depth adjustment
        # 3. Apply spread/condition multipliers
        # 4. Calculate prices and costs
        # 5. Generate execution recommendations
        ...

    def estimate_conservative(
        self,
        side: str,
        quantity: Decimal,
    ) -> ImpactEstimate:
        """Conservative fallback when no analysis available."""
        ...
```

### Key Methods

#### 1. `estimate()` - Main Impact Calculation
**Inputs:**
- `side`: "buy" or "sell"
- `quantity`: Order size (base currency)
- `analysis`: DepthAnalysis (from Phase 2)
- `volume_metrics`: Volume data (from Phase 1)

**Outputs:**
- `ImpactEstimate` dataclass with:
  - `estimated_impact_bps`
  - `estimated_avg_price`
  - `max_impact_price`
  - `slippage_cost`
  - `recommended_slicing` (bool)
  - `max_slice_size` (if slicing recommended)
  - `use_post_only` (bool)

**Algorithm:**
```python
# Step 1: Base impact (square-root model)
notional = quantity * mid_price
volume_15m = max(volume_metrics["volume_15m"], Decimal("1000"))
base_impact_bps = (notional / volume_15m).sqrt() * 100

# Step 2: Depth adjustment
if notional > analysis.depth_usd_5:
    depth_multiplier = (notional / analysis.depth_usd_5).sqrt()
    base_impact_bps *= depth_multiplier

# Step 3: Spread & condition adjustment
spread_multiplier = 1 + (analysis.spread_bps / 1000)
condition_multiplier = CONDITION_MAP[analysis.condition]  # 0.5x - 3.0x
final_impact_bps = base_impact_bps * spread_multiplier * condition_multiplier

# Step 4: Price calculation
if side == "buy":
    estimated_avg_price = mid_price * (1 + final_impact_bps / 10000)
else:
    estimated_avg_price = mid_price * (1 - final_impact_bps / 10000)

# Step 5: Execution recommendations
recommended_slicing = final_impact_bps > self.max_impact_bps
use_post_only = analysis.condition in [FAIR, POOR, CRITICAL] or final_impact_bps > max_impact_bps/2
```

#### 2. `estimate_conservative()` - Fallback Estimator
**When:** No DepthAnalysis available (missing orderbook data)

**Algorithm:**
```python
# Conservative assumptions
impact_bps = Decimal("100")  # 10bps
estimated_avg_price = Decimal("0")
recommended_slicing = True
max_slice_size = quantity / 10
use_post_only = True
```

#### 3. `_calculate_base_impact()` - Helper
Square-root model calculation isolated for testability.

#### 4. `_calculate_slicing_recommendation()` - Helper
Slicing logic (max slice size calculation) isolated.

---

## Test Plan (15-18 tests)

### Base Impact Calculation (3 tests)
1. `test_calculates_base_impact_square_root_model`
2. `test_base_impact_scales_with_notional`
3. `test_minimum_volume_floor_prevents_division_by_zero`

### Depth Adjustment (3 tests)
4. `test_applies_depth_multiplier_when_notional_exceeds_depth`
5. `test_no_depth_adjustment_when_notional_within_depth`
6. `test_depth_adjustment_uses_sqrt_multiplier`

### Spread & Condition Multipliers (4 tests)
7. `test_applies_spread_multiplier`
8. `test_excellent_condition_reduces_impact`
9. `test_critical_condition_increases_impact`
10. `test_combines_spread_and_condition_multipliers`

### Price Calculations (3 tests)
11. `test_buy_increases_price_by_impact`
12. `test_sell_decreases_price_by_impact`
13. `test_slippage_cost_calculation`

### Execution Recommendations (4 tests)
14. `test_recommends_slicing_when_impact_exceeds_threshold`
15. `test_calculates_max_slice_size_for_target_impact`
16. `test_recommends_post_only_for_poor_liquidity`
17. `test_recommends_post_only_for_high_impact`

### Edge Cases & Fallback (2-3 tests)
18. `test_conservative_estimate_when_no_analysis`
19. `test_handles_zero_volume_gracefully`
20. (Optional) `test_handles_extreme_notional_values`

---

## Integration Points

### LiquidityService Changes

**Before:**
```python
def estimate_market_impact(
    self,
    symbol: str,
    side: str,
    quantity: Decimal,
    book_data: tuple[...] | None = None,
) -> ImpactEstimate:
    # 110 lines of impact logic
    ...
```

**After:**
```python
def estimate_market_impact(
    self,
    symbol: str,
    side: str,
    quantity: Decimal,
    book_data: tuple[...] | None = None,
) -> ImpactEstimate:
    analysis = self._latest_analysis.get(symbol)
    if not analysis:
        return self._impact_estimator.estimate_conservative(side, quantity)

    volume_metrics = self._metrics_tracker.get_volume_metrics(symbol)
    return self._impact_estimator.estimate(side, quantity, analysis, volume_metrics)
```

### Constructor Updates

```python
def __init__(
    self,
    max_impact_bps: Decimal = Decimal("50"),
    depth_analysis_levels: int = 20,
    volume_window_minutes: int = 15,
    metrics_tracker: MetricsTracker | None = None,
    depth_analyzer: DepthAnalyzer | None = None,
    impact_estimator: ImpactEstimator | None = None,  # NEW
):
    # ...
    self._impact_estimator = impact_estimator or ImpactEstimator(
        max_impact_bps=max_impact_bps
    )
```

---

## Dependencies

### Inputs from Previous Phases

**Phase 1 (MetricsTracker):**
- `volume_metrics["volume_15m"]` - For base impact calculation

**Phase 2 (DepthAnalyzer):**
- `analysis.depth_usd_5` - For depth adjustment
- `analysis.spread_bps` - For spread multiplier
- `analysis.condition` - For condition multiplier
- `analysis.bid_price`, `analysis.ask_price` - For mid price

**LiquidityService:**
- `ImpactEstimate` dataclass (already defined)
- `LiquidityCondition` enum (already defined)

---

## Expected Outcomes

### Line Reduction
- **Before:** 407 lines (liquidity_service.py)
- **After:** ~290 lines (liquidity_service.py)
- **Reduction:** ~117 lines (-28.7%)
- **New module:** 140 lines (impact_estimator.py)

### Test Coverage
- **New tests:** 15-18 tests
- **Cumulative:** 77 → 92-95 tests
- **Zero regressions:** All existing liquidity tests must pass

### Code Quality
- ✅ Pure calculation (no state, easily testable)
- ✅ Dependency injection (ImpactEstimator can be mocked)
- ✅ Clear separation: Impact ≠ Depth ≠ Metrics
- ✅ Single Responsibility: Only market impact estimation

---

## Risk Assessment

**Low Risk** ✅:
- Pure calculation logic (stateless)
- Well-defined algorithm (square-root model)
- Existing tests provide regression safety
- Clear input/output contracts

**Medium Risk** ⚠️:
- Slicing calculation has complex threshold logic
- Condition multiplier mapping needs verification
- Conservative fallback values need validation

**Mitigation:**
- Comprehensive edge case tests (zero volume, extreme notional)
- Test each multiplier independently
- Verify conservative fallback aligns with risk tolerance

---

## Success Criteria

Before proceeding to Phase 4:
- [ ] ImpactEstimator created with ~140 lines
- [ ] 15-18 comprehensive unit tests (all passing)
- [ ] Integrated into LiquidityService with DI
- [ ] All 77 existing tests still passing
- [ ] Line reduction ≥ 25% (407 → ~290)

---

## Next Steps

1. **Review this plan** - Confirm scope and boundaries
2. **Implement ImpactEstimator** - Extract logic to new module
3. **Write 15-18 unit tests** - Cover all scenarios
4. **Integrate & verify** - Update LiquidityService, run all tests
5. **Phase 4 prep** - LiquidityScorer extraction ready to go

---

**Ready for approval?** Once approved, I'll extract ImpactEstimator following the established playbook.
