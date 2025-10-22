# Strategy Uplift and Validation Framework

## Overview

The strategy uplift framework validates that enhanced strategies beat the baseline MA crossover across different market conditions using robust out-of-sample testing.

**Phase 2 Exit Criteria**:
- Enhanced strategy must beat baseline in **ALL 3 regimes** (trend/range/high-vol)
- Enhanced strategy must win **≥60% of out-of-sample CV splits**

---

## Why Validate Strategy Uplift?

Before deploying a more complex strategy, you must prove it's actually better than the simple baseline:

1. **Avoid False Improvements**: A strategy that only works in-sample is overfitted
2. **Regime Robustness**: Must work in trends, ranges, and high volatility
3. **Out-of-Sample Validation**: Must generalize to unseen data
4. **Risk Management**: Don't add complexity without proven benefit

**The validation framework ensures** your enhanced strategy is genuinely better, not just curve-fitted to historical data.

---

## Quick Start

```python
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.features.live_trade.strategies.enhanced_strategy import EnhancedStrategy, EnhancedStrategyConfig
from bot_v2.features.optimize.strategy_comparison import StrategyComparator

# Create baseline
baseline = BaselinePerpsStrategy(config=baseline_config)

# Create enhanced with filters
enhanced_config = EnhancedStrategyConfig(
    # ... baseline params ...
    enable_volatility_filter=True,
    enable_trend_filter=True,
    adx_min_threshold=20.0,
)
enhanced = EnhancedStrategy(config=enhanced_config)

# Compare
comparator = StrategyComparator(
    baseline_strategy=baseline,
    enhanced_strategy=enhanced,
)

# Run validations
full_result = comparator.compare_full_dataset(data=data, symbol="BTC-PERP")
regime_results = comparator.compare_by_regime(data=data, symbol="BTC-PERP")
cv_results = comparator.compare_walk_forward(data=data, symbol="BTC-PERP", n_splits=5)

# Generate report
print(comparator.generate_report(
    full_result=full_result,
    regime_results=regime_results,
    cv_results=cv_results,
))
```

---

## Enhanced Strategy Components

### 1. Volatility Filter

**Purpose**: Avoid trading in extreme volatility (too high or too low)

**Implementation**:
- Calculate ATR (Average True Range) over 14 periods
- Block trades if `ATR > 2.0x average` (too volatile)
- Block trades if `ATR < 0.5x average` (too quiet, likely ranging)

**Rationale**:
- High volatility → whipsaws, false breakouts, wide slippage
- Low volatility → ranging market, MA crossovers unreliable

```python
enhanced_config = EnhancedStrategyConfig(
    enable_volatility_filter=True,
    atr_period=14,
    atr_high_threshold_multiplier=2.0,  # Block if ATR > 2x average
    atr_low_threshold_multiplier=0.5,   # Block if ATR < 0.5x average
)
```

### 2. Trend Strength Filter

**Purpose**: Only take MA crossover signals in trending markets

**Implementation**:
- Calculate ADX (Average Directional Index) over 14 periods
- Require `ADX ≥ 20` for new entries
- If `ADX < 20`, market is choppy/ranging → skip signal

**Rationale**:
- MA crossovers work in trends, fail in ranges
- ADX < 20 = weak/no trend (choppy market)
- ADX ≥ 20 = trend developing (crossover more reliable)

```python
enhanced_config = EnhancedStrategyConfig(
    enable_trend_filter=True,
    adx_period=14,
    adx_min_threshold=20.0,  # Require ADX >= 20
)
```

### 3. Dynamic Stops (Future)

**Purpose**: Adjust stop-loss width based on current volatility

**Implementation** (not enabled in Phase 2):
- Calculate stop distance as `N × ATR` (e.g., 2× ATR)
- Wider stops in volatile periods (avoid premature stops)
- Tighter stops in calm periods (preserve capital)

**Rationale**:
- Fixed 2% stop may be too tight in high vol, too wide in low vol
- ATR-based stops adapt to current market conditions

```python
enhanced_config = EnhancedStrategyConfig(
    enable_dynamic_stops=True,  # Not recommended for Phase 2
    atr_stop_multiplier=2.0,
    min_stop_pct=0.005,  # 0.5% minimum
    max_stop_pct=0.05,   # 5% maximum
)
```

---

## Market Regime Detection

### What Are Market Regimes?

Markets exhibit different behaviors that require different strategy approaches:

| Regime | Characteristics | Best Strategy |
|--------|----------------|---------------|
| **TREND** | Strong directional movement (ADX > 25) | MA crossover works well |
| **RANGE** | Sideways/choppy (ADX < 20) | MA crossover fails (many whipsaws) |
| **HIGH_VOL** | High volatility (ATR > 1.5x avg) | Wide stops needed, fewer trades |

### Regime Detection

```python
from bot_v2.features.optimize.regime_detection import RegimeDetector, split_data_by_regime

# Create detector
detector = RegimeDetector(
    adx_trend_threshold=25.0,    # ADX > 25 = trending
    adx_range_threshold=20.0,     # ADX < 20 = ranging
    atr_volatility_multiplier=1.5,  # ATR > 1.5x avg = high vol
    min_regime_bars=20,           # Minimum 20 bars per regime
)

# Detect regimes
regimes = detector.detect_regimes(data)

# Split data by regime
regime_data = split_data_by_regime(data, detector)

# Now you have:
regime_data[MarketRegime.TREND]     # Trending periods only
regime_data[MarketRegime.RANGE]     # Ranging periods only
regime_data[MarketRegime.HIGH_VOL]  # High volatility periods only
```

### Why Regime Validation Matters

A strategy that only works in trends is **not robust**:
- Real markets cycle through all three regimes
- Must prove strategy works (or doesn't fail) in each
- Enhanced filters should **prevent bad trades in ranges/high-vol**

**Phase 2 Requirement**: Enhanced must beat baseline in ALL 3 regimes.

---

## Out-of-Sample Validation

### The Overfitting Problem

**Overfitting**: Strategy parameters that work perfectly in-sample but fail on new data.

**Example**:
- You optimize MA periods to (7, 23) because it has best Sharpe on 2023 data
- Deploy in 2024 → loses money (parameters were curve-fitted to noise)

**Solution**: Out-of-sample validation with purged CV.

### Purged Walk-Forward Cross-Validation

**Concept**: Test strategy on data it has never seen, with embargo periods to prevent leakage.

**Timeline**:
```
[-----Train 1-----] [Test 1] [Embargo]
[---------Train 2---------] [Test 2] [Embargo]
[---------------Train 3---------------] [Test 3] [Embargo]
```

**Key features**:
1. **Temporal ordering**: Training always precedes test (no look-ahead)
2. **Embargo**: Buffer period after test prevents using future data
3. **Multiple splits**: Validates across different time periods

```python
from bot_v2.features.optimize.purged_cv import AnchoredWalkForwardCV

cv = AnchoredWalkForwardCV(
    n_splits=5,        # 5 train/test splits
    embargo_pct=0.02,  # 2% embargo (e.g., 2% of 100 bars = 2 bar buffer)
)

splits = cv.split(data)

for split in splits:
    train_data = split.get_train_data(data)
    test_data = split.get_test_data(data)
    # Run backtest on test_data only
```

**Phase 2 Requirement**: Enhanced must win ≥60% of CV splits.

---

## Strategy Comparison Framework

### StrategyComparator

Compares baseline vs enhanced across multiple dimensions:

```python
from bot_v2.features.optimize.strategy_comparison import StrategyComparator

comparator = StrategyComparator(
    baseline_strategy=baseline,
    enhanced_strategy=enhanced,
    backtest_config=backtest_config,
)
```

### 1. Full Dataset Comparison

```python
result = comparator.compare_full_dataset(data=data, symbol="BTC-PERP")

print(f"Baseline: {result.baseline_performance.sharpe_ratio:.2f}")
print(f"Enhanced: {result.enhanced_performance.sharpe_ratio:.2f}")
print(f"Winner: {result.enhanced_wins()}")
```

### 2. Regime-Specific Comparison

```python
regime_results = comparator.compare_by_regime(data=data, symbol="BTC-PERP")

for regime, result in regime_results.items():
    print(f"{regime}: Enhanced wins = {result.enhanced_wins()}")
```

**Exit Criterion**: Enhanced must win **ALL** regimes.

### 3. Walk-Forward CV Comparison

```python
cv_results = comparator.compare_walk_forward(data=data, symbol="BTC-PERP", n_splits=5)

wins = sum(1 for r in cv_results if r.enhanced_wins())
print(f"Enhanced wins {wins}/{len(cv_results)} splits")
```

**Exit Criterion**: Enhanced must win **≥60%** of splits.

### 4. Generate Report

```python
report = comparator.generate_report(
    full_result=full_result,
    regime_results=regime_results,
    cv_results=cv_results,
)
print(report)
```

**Report includes**:
- Full dataset metrics
- Regime-specific breakdowns
- CV split results
- Phase 2 exit criteria pass/fail

---

## Performance Metrics

### StrategyPerformance

Each strategy is evaluated on:

| Metric | Description | Target |
|--------|-------------|--------|
| **Total Return** | Overall P&L % | Higher is better |
| **Sharpe Ratio** | Risk-adjusted return | > 1.0 good, > 2.0 excellent |
| **Max Drawdown** | Largest peak-to-trough decline | Lower is better (< 30%) |
| **Win Rate** | % of profitable trades | > 50% |
| **Calmar Ratio** | Return / Max DD | Higher is better |
| **Total Trades** | Number of round-trip trades | - |

### Composite Score

Strategies are ranked using weighted composite score:
- Sharpe ratio: 40%
- Total return: 30%
- Max drawdown: 20% (negative weight)
- Win rate: 10%

**Enhanced wins if**: `enhanced_score > baseline_score`

---

## Configuration Guide

### Conservative Enhancement (Recommended for Phase 2)

```python
EnhancedStrategyConfig(
    # Baseline params (same as baseline strategy)
    short_ma_period=5,
    long_ma_period=20,
    position_fraction=0.20,
    target_leverage=3,
    trailing_stop_pct=0.02,

    # Enhancement params (conservative)
    enable_volatility_filter=True,
    atr_high_threshold_multiplier=2.0,  # Block extreme vol
    atr_low_threshold_multiplier=0.5,   # Block dead markets

    enable_trend_filter=True,
    adx_min_threshold=20.0,  # Moderate threshold

    enable_dynamic_stops=False,  # Keep simple for Phase 2
)
```

### Aggressive Enhancement (Phase 3)

```python
EnhancedStrategyConfig(
    # ... baseline params ...

    # Tighter filters
    atr_high_threshold_multiplier=1.5,  # Stricter vol filter
    adx_min_threshold=25.0,  # Require stronger trend

    # Dynamic stops
    enable_dynamic_stops=True,
    atr_stop_multiplier=2.5,
)
```

---

## Troubleshooting

### Enhanced Loses in Trend Regime

**Symptoms**: Baseline wins in TREND, enhanced loses

**Root Causes**:
1. ADX threshold too high (blocking valid trend trades)
2. Volatility filter too strict

**Fixes**:
```python
# Lower ADX threshold
config.adx_min_threshold = 15.0  # was 20.0

# Widen volatility range
config.atr_high_threshold_multiplier = 2.5  # was 2.0
```

### Enhanced Loses in Range Regime

**Symptoms**: Enhanced loses less than baseline, but still negative

**Root Causes**:
1. Filters not strict enough (still trading in choppy market)
2. Range regime has weak trends (ADX ~15-20)

**Fixes**:
```python
# Tighten trend filter
config.adx_min_threshold = 25.0  # was 20.0

# Add low-vol filter
config.enable_volatility_filter = True
config.atr_low_threshold_multiplier = 0.6  # Block quiet markets
```

**Note**: It's OK if both strategies lose in range regimes. Enhanced just needs to lose **less**.

### Enhanced Loses Out-of-Sample

**Symptoms**: Enhanced wins in-sample but loses in CV splits

**Root Causes**:
1. **Overfitting**: Parameters optimized for specific period
2. **Regime shift**: In-sample data not representative

**Fixes**:
```python
# Use simpler filters
config.enable_dynamic_stops = False  # Remove complexity

# Wider filter thresholds (more conservative)
config.atr_high_threshold_multiplier = 2.5  # was 2.0
config.adx_min_threshold = 15.0  # was 20.0

# Re-run with different CV splits
cv = AnchoredWalkForwardCV(n_splits=10)  # was 5
```

### Enhanced Wins < 60% of CV Splits

**Symptoms**: Enhanced wins 2-3 out of 5 splits (< 60%)

**Acceptable scenarios**:
- If enhanced wins by large margin in winning splits
- If baseline also fails in same splits (regime issue)

**Fixes**:
1. Analyze losing splits to find pattern
2. Adjust filters to handle those periods
3. Consider accepting if average improvement > 20%

---

## Integration with Phase 2 Workflow

### Complete Phase 2 Validation

```
✓ Workstream 1: Stress Testing
  - Run derivatives stress tests
  - Ensure 0 liquidations, < 30% drawdown

✓ Workstream 2: INTX Eligibility
  - Verify API entitlements
  - Test fail-closed logic

✓ Workstream 3: Strategy Uplift (THIS)
  - Enhanced beats baseline in ALL regimes
  - Enhanced wins ≥60% of CV splits

→ Phase 2 Complete → Ready for Phase 3
```

### Full Validation Script

```python
# 1. Stress testing
from bot_v2.features.optimize.stress_validator import StressTestValidator

validator = StressTestValidator(strategy=enhanced_strategy, criteria=criteria)
validator.run_standard_suite(data=data, symbol="BTC-PERP")

if not validator.passed_all():
    print("❌ Failed stress tests")
    exit(1)

# 2. Strategy comparison
comparator = StrategyComparator(baseline_strategy=baseline, enhanced_strategy=enhanced)
regime_results = comparator.compare_by_regime(data=data, symbol="BTC-PERP")
cv_results = comparator.compare_walk_forward(data=data, symbol="BTC-PERP")

regime_wins = sum(1 for r in regime_results.values() if r.enhanced_wins())
cv_wins = sum(1 for r in cv_results if r.enhanced_wins())

if regime_wins < len(regime_results) or cv_wins < len(cv_results) * 0.6:
    print("❌ Failed strategy uplift validation")
    exit(1)

print("✅ ALL PHASE 2 CRITERIA PASSED")
```

---

## Advanced Topics

### Custom Performance Metrics

```python
class CustomPerformance(StrategyPerformance):
    def score(self) -> float:
        # Custom weighting
        return (
            self.sharpe_ratio * 0.5 +
            self.calmar_ratio * 0.3 +
            self.total_return * 0.2
        )
```

### Regime-Aware Parameter Tuning

```python
# Detect regimes first
regime_data = split_data_by_regime(data)

# Test different params per regime
if len(regime_data[MarketRegime.RANGE]) > 100:
    # Tighten filters for range
    config.adx_min_threshold = 25.0
else:
    # Relax for trend-heavy data
    config.adx_min_threshold = 15.0
```

### Multi-Symbol Validation

```python
symbols = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]

for symbol in symbols:
    result = comparator.compare_full_dataset(data=data[symbol], symbol=symbol)
    print(f"{symbol}: Enhanced wins = {result.enhanced_wins()}")
```

---

## See Also

- [Production-Parity Backtesting](./production_parity_backtesting.md)
- [Derivatives Stress Testing](../derivatives/stress_testing.md)
- [INTX Eligibility](../derivatives/intx_eligibility.md)
- [Regime Detection Reference](../reference/regime_detection.md)

---

## Appendix: Research References

### Overfitting in Finance

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 7: Cross-Validation in Finance
  - Chapter 12: Backtesting through Cross-Validation

### Market Regimes

- Kritzman, M., & Li, Y. (2010). "Skulls, Financial Turbulence, and Risk Management." *Financial Analysts Journal*, 66(5), 30-41.

### Technical Indicators

- Wilder, J.W. (1978). *New Concepts in Technical Trading Systems*.
  - ADX and ATR definitions
