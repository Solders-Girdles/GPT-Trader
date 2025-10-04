# LiquidityService Refactor Plan

**Target Module:** `src/bot_v2/features/live_trade/liquidity_service.py`
**Current Size:** 576 lines
**Complexity:** High (order book analysis, impact modeling, liquidity scoring)
**Pattern:** Extract → Test → Compose (following StrategyOrchestrator playbook)

## Overview

Extract specialized services from LiquidityService to improve testability, maintainability, and separation of concerns. The service currently handles 4 distinct responsibilities that can be cleanly separated.

## Current Architecture

```
liquidity_service.py (576 lines)
├── LiquidityMetrics class (~85 lines)
│   ├── Trade windowing (15-minute rolling window)
│   ├── Spread tracking
│   ├── Volume metrics aggregation
│   └── Old data cleanup
│
├── LiquidityService.analyze_order_book() (~115 lines)
│   ├── Level 1 data extraction (best bid/ask)
│   ├── Spread calculation (absolute & bps)
│   ├── Depth calculation (1%, 5%, 10% levels)
│   ├── Imbalance metrics (bid/ask ratio, depth imbalance)
│   ├── Liquidity scoring (composite score 0-100)
│   └── Condition determination (EXCELLENT → CRITICAL)
│
├── LiquidityService.estimate_market_impact() (~110 lines)
│   ├── Square-root impact model
│   ├── Depth-adjusted multipliers
│   ├── Spread and condition adjustments
│   ├── Slippage cost calculation
│   ├── Execution recommendations (slicing, post-only)
│   └── Conservative fallback estimates
│
└── Scoring methods (~40 lines)
    ├── _score_spread() - Spread quality scoring
    ├── _score_depth() - Depth adequacy scoring
    ├── _score_imbalance() - Imbalance penalty
    └── _determine_condition() - Map score → LiquidityCondition
```

## Proposed Architecture

```
LiquidityService (facade, ~200 lines)
├── MetricsTracker (~100 lines) - NEW MODULE
│   ├── Trade windowing
│   ├── Spread tracking
│   ├── Volume metrics
│   └── Time-series cleanup
│
├── DepthAnalyzer (~150 lines) - NEW MODULE
│   ├── Level 1 extraction
│   ├── Spread calculation
│   ├── Multi-level depth calculation (1%, 5%, 10%)
│   ├── Imbalance metrics
│   └── Depth aggregation helpers
│
├── ImpactEstimator (~140 lines) - NEW MODULE
│   ├── Square-root model
│   ├── Depth adjustment
│   ├── Spread/condition multipliers
│   ├── Slippage calculation
│   ├── Execution recommendations
│   └── Conservative fallback logic
│
└── LiquidityScorer (~80 lines) - NEW MODULE
    ├── Spread scoring (bps → 0-100)
    ├── Depth scoring (USD depth → 0-100)
    ├── Imbalance penalty
    ├── Composite score calculation
    └── Condition mapping
```

## Phased Extraction Plan

### Phase 1: MetricsTracker (Est. ~100 lines)
**Target:** Extract `LiquidityMetrics` to standalone module

**New file:** `src/bot_v2/features/live_trade/liquidity_metrics_tracker.py`

**Extracted:**
- `LiquidityMetrics` class
- Trade windowing logic
- Spread tracking
- Volume aggregation
- Old data cleanup

**Tests:** 10-12 unit tests
- Trade addition and windowing
- Spread tracking
- Volume metrics calculation (15m, avg, peak)
- Time-based cleanup

**Line reduction:** 576 → ~490 lines

**Integration:**
```python
class LiquidityService:
    def __init__(self, ...):
        self._metrics_tracker = MetricsTracker(window_minutes=15)

    def update_trade_data(self, symbol: str, price: Decimal, size: Decimal):
        self._metrics_tracker.add_trade(symbol, price, size)
```

---

### Phase 2: DepthAnalyzer (Est. ~150 lines)
**Target:** Extract order book depth analysis

**New file:** `src/bot_v2/features/live_trade/depth_analyzer.py`

**Extracted:**
- Level 1 data extraction (best bid/ask)
- Spread calculation (absolute + bps)
- Multi-level depth calculation (_calculate_depth logic)
- Depth at 1%, 5%, 10% thresholds
- Imbalance metrics (bid/ask ratio, depth imbalance)

**Tests:** 12-15 unit tests
- Empty book handling
- Level 1 extraction
- Spread calculation (absolute, bps, zero-price edge case)
- Depth calculation at multiple levels
- Imbalance metrics
- Edge cases (missing bids/asks, zero depth)

**Line reduction:** ~490 → ~340 lines

**Integration:**
```python
class LiquidityService:
    def __init__(self, ...):
        self._depth_analyzer = DepthAnalyzer(depth_levels=20)

    def analyze_order_book(self, symbol, bids, asks, timestamp):
        depth_data = self._depth_analyzer.analyze_depth(
            bids, asks, thresholds=[0.01, 0.05, 0.10]
        )
        # Use depth_data for scoring
```

---

### Phase 3: ImpactEstimator (Est. ~140 lines)
**Target:** Extract market impact estimation

**New file:** `src/bot_v2/features/live_trade/impact_estimator.py`

**Extracted:**
- Square-root impact model
- Depth-based multipliers
- Spread/condition adjustments
- Slippage cost calculation
- Execution recommendations (slicing thresholds, post-only logic)
- Conservative fallback estimates

**Tests:** 15-18 unit tests
- Base impact calculation (square-root model)
- Depth adjustment multipliers
- Spread adjustments
- Condition-based multipliers
- Buy vs sell price impact
- Slippage cost calculation
- Slicing recommendations
- Post-only recommendations
- Conservative fallback (no depth data)
- Edge cases (zero volume, extreme notional)

**Line reduction:** ~340 → ~200 lines

**Integration:**
```python
class LiquidityService:
    def __init__(self, max_impact_bps=Decimal("50"), ...):
        self._impact_estimator = ImpactEstimator(
            max_impact_bps=max_impact_bps
        )

    def estimate_market_impact(self, symbol, side, quantity, book_data):
        analysis = self._latest_analysis[symbol]
        volume_metrics = self._metrics_tracker.get_volume_metrics(symbol)
        return self._impact_estimator.estimate(
            side=side,
            quantity=quantity,
            analysis=analysis,
            volume_metrics=volume_metrics,
        )
```

---

### Phase 4: LiquidityScorer (Est. ~80 lines)
**Target:** Extract liquidity scoring logic

**New file:** `src/bot_v2/features/live_trade/liquidity_scorer.py`

**Extracted:**
- _score_spread() - Spread quality (bps → 0-100)
- _score_depth() - Depth adequacy (USD depth → 0-100)
- _score_imbalance() - Imbalance penalty
- Composite score calculation
- _determine_condition() - Score → LiquidityCondition mapping

**Tests:** 12-15 unit tests
- Spread scoring (tight, normal, wide spreads)
- Depth scoring (deep, shallow markets)
- Imbalance penalty (balanced, skewed, extreme)
- Composite score calculation
- Condition determination (EXCELLENT → CRITICAL thresholds)
- Edge cases (zero depth, extreme spread)

**Line reduction:** ~200 → ~150 lines (LiquidityService becomes lean facade)

**Integration:**
```python
class LiquidityService:
    def __init__(self, ...):
        self._liquidity_scorer = LiquidityScorer()

    def analyze_order_book(self, ...):
        depth_data = self._depth_analyzer.analyze_depth(...)
        score = self._liquidity_scorer.calculate_score(
            spread_bps=depth_data.spread_bps,
            depth_usd_1=depth_data.depth_usd_1,
            depth_usd_5=depth_data.depth_usd_5,
            imbalance=depth_data.depth_imbalance,
            mid_price=depth_data.mid_price,
        )
        condition = self._liquidity_scorer.determine_condition(score)
```

---

### Phase 5: Integration & Cleanup
**Activities:**
1. Verify all extracted services work together
2. Add end-to-end characterization tests
3. Update documentation
4. Clean up any remaining LiquidityService methods
5. Ensure backward compatibility

**Characterization Tests:** 5-7 tests
- Service initialization creates all sub-services
- analyze_order_book uses depth analyzer and scorer
- estimate_market_impact uses impact estimator and metrics
- Full flow: update trade → analyze book → estimate impact
- Snapshot generation uses all services

**Final Metrics:**
- **Line reduction:** 576 → ~150 lines (-73% reduction)
- **Extracted code:** ~470 lines (4 new services)
- **Test coverage:** +50-60 new unit tests

---

## Component Responsibilities

### LiquidityService (Facade)
- **Role:** Orchestrate liquidity analysis workflow
- **Delegates to:**
  - `metrics_tracker.add_trade()` for trade updates
  - `depth_analyzer.analyze_depth()` for order book analysis
  - `liquidity_scorer.calculate_score()` for scoring
  - `impact_estimator.estimate()` for impact calculation
- **Maintains:**
  - Per-symbol latest analysis cache
  - Service instance management
  - Snapshot generation (combines all data)

### MetricsTracker
- **Responsibilities:**
  - Track trades within rolling time window
  - Calculate volume metrics (15m volume, avg trade, peak)
  - Track spread over time
  - Clean expired data
- **Key Methods:**
  - `add_trade(symbol, price, size, timestamp)` - Add trade to window
  - `add_spread(symbol, spread_bps, timestamp)` - Track spread
  - `get_volume_metrics(symbol)` - Return volume stats
  - `_clean_old_data()` - Remove expired entries

### DepthAnalyzer
- **Responsibilities:**
  - Extract Level 1 data (best bid/ask)
  - Calculate spread (absolute + bps)
  - Compute depth at multiple price levels (1%, 5%, 10%)
  - Calculate imbalance metrics
- **Key Methods:**
  - `analyze_depth(bids, asks, thresholds)` - Main analysis
  - `calculate_depth_at_level(levels, min_price, max_price)` - Depth helper
  - `extract_level1(bids, asks)` - Best bid/ask
  - `calculate_imbalance(bid_depth, ask_depth)` - Imbalance metrics

### ImpactEstimator
- **Responsibilities:**
  - Estimate market impact using square-root model
  - Apply depth, spread, and condition adjustments
  - Calculate slippage costs
  - Generate execution recommendations (slicing, post-only)
- **Key Methods:**
  - `estimate(side, quantity, analysis, volume_metrics)` - Main estimation
  - `calculate_base_impact(notional, volume)` - Square-root model
  - `apply_depth_adjustment(impact, notional, depth)` - Depth multiplier
  - `recommend_execution(impact, condition)` - Slicing/post-only logic

### LiquidityScorer
- **Responsibilities:**
  - Score spread quality (0-100)
  - Score depth adequacy (0-100)
  - Apply imbalance penalties
  - Calculate composite liquidity score
  - Map score to LiquidityCondition
- **Key Methods:**
  - `calculate_score(spread_bps, depth_usd_1, depth_usd_5, imbalance, mid_price)` - Composite
  - `score_spread(spread_bps)` - Spread component
  - `score_depth(depth_usd, mid_price)` - Depth component
  - `score_imbalance(imbalance)` - Imbalance penalty
  - `determine_condition(score)` - Score → EXCELLENT/GOOD/FAIR/POOR/CRITICAL

---

## Design Patterns

### Dependency Injection
All services support constructor injection:
```python
class LiquidityService:
    def __init__(
        self,
        metrics_tracker: MetricsTracker | None = None,
        depth_analyzer: DepthAnalyzer | None = None,
        impact_estimator: ImpactEstimator | None = None,
        liquidity_scorer: LiquidityScorer | None = None,
        max_impact_bps: Decimal = Decimal("50"),
        depth_analysis_levels: int = 20,
    ):
        self._metrics_tracker = metrics_tracker or MetricsTracker()
        self._depth_analyzer = depth_analyzer or DepthAnalyzer(depth_levels=depth_analysis_levels)
        self._impact_estimator = impact_estimator or ImpactEstimator(max_impact_bps=max_impact_bps)
        self._liquidity_scorer = liquidity_scorer or LiquidityScorer()
```

### Stateless Services
All extracted services are stateless (except MetricsTracker which manages time-series data):
- DepthAnalyzer: Pure calculation, no state
- ImpactEstimator: Pure calculation, no state
- LiquidityScorer: Pure calculation, no state
- MetricsTracker: Manages per-symbol time-series windows

### Data Transfer Objects
Use existing dataclasses for clean interfaces:
- `DepthAnalysis` - Output from depth analysis
- `ImpactEstimate` - Output from impact estimation
- Volume metrics dict - Output from metrics tracker

---

## Success Criteria

Before each phase:
- [ ] All existing tests pass (no regressions)
- [ ] New service has 10+ comprehensive unit tests
- [ ] Integration with LiquidityService verified
- [ ] Characterization tests updated if needed

Final acceptance:
- [ ] Line reduction ≥ 60% (576 → ~200 lines)
- [ ] Test coverage +50-60 tests
- [ ] All orchestration tests pass
- [ ] End-to-end characterization tests pass
- [ ] Documentation complete

---

## Risk Assessment

**Low Risk** ✅:
- Clean separation of concerns (depth, impact, scoring)
- Existing test coverage provides regression protection
- Stateless design simplifies testing
- Clear data flow (metrics → depth → score → impact)

**Medium Risk** ⚠️:
- Impact estimation has complex multiplier logic
- Square-root model needs careful extraction
- Volume metrics integration requires coordination

**High Risk** ❌:
- None identified

---

## Related Work

- **Similar refactors:**
  - StrategyOrchestrator (411 → 332 lines, 4 extractions)
  - BackupManager (636 → 431 lines, 3 extractions)

- **Documentation:**
  - `STRATEGY_ORCHESTRATOR_REFACTOR.md` - Similar pattern
  - `BACKUP_OPERATIONS_REFACTOR.md` - Facade pattern example

---

## Timeline Estimate

| Phase | LOC | Tests | Estimate |
|-------|-----|-------|----------|
| Phase 1: MetricsTracker | ~100 | 10-12 | 6-8h |
| Phase 2: DepthAnalyzer | ~150 | 12-15 | 8-10h |
| Phase 3: ImpactEstimator | ~140 | 15-18 | 10-12h |
| Phase 4: LiquidityScorer | ~80 | 12-15 | 6-8h |
| Phase 5: Integration | N/A | 5-7 | 4-6h |
| **Total** | **~470** | **54-67** | **34-44h** |

With 25% buffer: **43-55 hours** total

---

## Next Steps

1. ✅ Draft extraction plan (this document)
2. ⏳ Review plan with team
3. ⏳ Create Phase 1 task: Extract MetricsTracker
4. ⏳ Begin Phase 1 extraction following playbook:
   - Extract code
   - Write 10-12 unit tests
   - Integrate into LiquidityService
   - Verify all tests pass
   - Update documentation

---

**Status:** 📋 Plan Draft
**Owner:** TBD
**Target Start:** TBD
**Target Completion:** TBD (estimated 1-2 weeks with dedicated effort)
