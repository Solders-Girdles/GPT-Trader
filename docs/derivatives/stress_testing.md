# Derivatives Stress Testing Framework

## Overview

The derivatives stress testing framework validates system resilience before live derivatives deployment by simulating extreme market conditions including gap moves, high volatility, funding rate shocks, and liquidity crises.

**Phase 2 Exit Criteria**: All stress tests must pass before enabling derivatives trading.

---

## Why Stress Testing?

Derivatives trading with leverage introduces risks not present in spot trading:

1. **Liquidation Risk**: Position can be force-closed if price moves against you
2. **Funding Costs**: Perpetuals incur 8-hourly funding payments
3. **Margin Requirements**: Must maintain minimum margin or face liquidation
4. **Gap Moves**: Overnight gaps can bypass stop losses
5. **Volatility Spikes**: High vol can exhaust margin quickly

**Stress testing validates** that your strategy and risk parameters can survive these scenarios without catastrophic losses or liquidations.

---

## Quick Start

```python
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.features.optimize.stress_validator import StressTestValidator, StressTestCriteria
from bot_v2.features.optimize.types_v2 import BacktestConfig

# Configure strategy
strategy = BaselinePerpsStrategy(config=your_config)

# Set stress test criteria
criteria = StressTestCriteria(
    max_liquidations_allowed=0,  # MUST avoid liquidation
    max_drawdown_pct=0.30,       # Max 30% drawdown
    max_leverage=10.0,            # Max 10x leverage
)

# Create validator
validator = StressTestValidator(
    strategy=strategy,
    criteria=criteria,
    backtest_config=BacktestConfig(initial_capital=Decimal("50000")),
)

# Run standard test suite
results = validator.run_standard_suite(data=historical_data, symbol="BTC-PERP")

# Check if passed
if validator.passed_all():
    print("✅ Ready for derivatives!")
else:
    print("❌ Failed - adjust parameters")
    print(validator.generate_report())
```

---

## Stress Test Scenarios

### 1. Gap Moves

**What**: Price jumps instantly without intermediate prices (5-20% gaps)

**Real-world causes**:
- Market closure → opening with overnight news
- Flash crashes (e.g., 2010 Flash Crash, 2017 ETH flash crash)
- Circuit breaker triggers
- Exchange outages

**Tests**:
- Gap down 5% (3 gaps over backtest period)
- Gap up 5% (3 gaps)

**Validates**:
- Stop losses still protect positions
- No liquidations from sudden moves
- System can handle discontinuous prices

### 2. High Volatility

**What**: 2-3x normal volatility while maintaining trend direction

**Real-world causes**:
- Earnings announcements
- Fed decisions
- Black swan events (COVID crash, LUNA collapse)
- Market panic/euphoria

**Tests**:
- 2x volatility over 30% of backtest period

**Validates**:
- Doesn't over-trade in choppy markets
- Trailing stops don't get stopped out prematurely
- Margin requirements don't spike unexpectedly

### 3. Flash Crash

**What**: Rapid 20% drop followed by 50% recovery

**Real-world examples**:
- 2010 Flash Crash (S&P down 9%, recovered in minutes)
- 2017 ETH flash crash (ETH dropped to $0.10 on GDAX)
- 2021 leveraged token liquidations

**Tests**:
- 20% drop over 5 bars, 50% recovery over 20 bars

**Validates**:
- System survives extreme drawdown
- Doesn't get liquidated during crash
- Can recover if market recovers

### 4. Funding Rate Shock

**What**: Funding rate spikes from 0.01% to 2% per 8h interval

**Real-world causes**:
- Extreme basis divergence (futures > spot)
- Liquidation cascades (2021 leveraged long squeeze)
- Exchange manipulation

**Tests**:
- Normal 0.01% funding → 2% shock for 24 hours

**Validates**:
- Funding costs don't eat all profits
- System closes positions if funding too high
- Long-term viability under adverse funding

### 5. Liquidity Crisis

**What**: 5x increased slippage and wider spreads

**Real-world causes**:
- Exchange technical issues
- Market makers pull liquidity
- Extreme vol events (market makers widen spreads)
- Low liquidity hours (overnight, weekends)

**Tests**:
- 5x slippage, 3x spreads for 20% of period

**Validates**:
- Strategy still profitable with bad execution
- Risk controls still work
- Can exit positions even with high slippage

---

## Exit Criteria

### Phase 2 Go/No-Go Checklist

| Criterion | Requirement | Why It Matters |
|-----------|-------------|----------------|
| **Liquidations** | 0 allowed | Liquidation = total position loss + fees |
| **Max Drawdown** | < 30% | Must survive worst-case scenario |
| **Leverage** | < 10x | Higher leverage = higher liquidation risk |
| **Margin Util** | < 90% | Need buffer for unexpected moves |
| **Funding Cost** | < 10% of P&L (optional) | High funding erodes profits |

**All criteria must pass for Phase 2 exit.**

---

## Configuration

### StressTestCriteria

```python
@dataclass
class StressTestCriteria:
    # Liquidation criteria
    max_liquidations_allowed: int = 0  # MUST be 0 for production
    max_liquidation_warnings: int | None = None  # Optional warning threshold

    # Drawdown criteria
    max_drawdown_pct: float = 0.30  # 30% max drawdown
    max_drawdown_warning_pct: float = 0.20  # Warning at 20%

    # Funding cost criteria
    max_funding_cost_pct_of_pnl: float | None = None  # e.g., 0.10 = 10% of P&L

    # Margin criteria
    max_margin_utilization: float = 0.90  # Max 90% margin utilization
    max_leverage: float = 10.0  # Max 10x leverage

    # Performance criteria (optional)
    min_sharpe_ratio: float | None = None
    max_negative_return_pct: float | None = None  # e.g., -0.50 = max 50% loss
```

### Recommended Settings by Risk Tolerance

**Conservative (Recommended for Phase 2)**:
```python
StressTestCriteria(
    max_liquidations_allowed=0,
    max_drawdown_pct=0.20,  # 20% max
    max_leverage=5.0,        # 5x max
    max_margin_utilization=0.70,  # 70% max
)
```

**Moderate**:
```python
StressTestCriteria(
    max_liquidations_allowed=0,
    max_drawdown_pct=0.30,  # 30% max
    max_leverage=10.0,       # 10x max
    max_margin_utilization=0.90,  # 90% max
)
```

**Aggressive (NOT recommended for initial deployment)**:
```python
StressTestCriteria(
    max_liquidations_allowed=1,  # Allow 1 liquidation
    max_drawdown_pct=0.50,  # 50% max
    max_leverage=20.0,       # 20x max
    max_margin_utilization=0.95,  # 95% max
)
```

---

## Advanced Usage

### Custom Scenarios

```python
from bot_v2.features.optimize.stress_scenarios import GapMoveScenario, HighVolatilityScenario

# Create custom gap scenario
custom_gap = GapMoveScenario(
    gap_size_pct=0.10,  # 10% gap
    gap_direction="down",
    num_gaps=5,
)

# Run custom test
validator.run_stress_test(
    scenario_name="Custom 10% Gap",
    scenario_type=StressScenarioType.GAP_MOVE,
    data=data,
    symbol="BTC-PERP",
    scenario_generator=custom_gap,
)
```

### Analyzing Individual Results

```python
for result in validator.results:
    print(f"Scenario: {result.scenario_name}")
    print(f"Status: {result.status.value}")

    # Access backtest result
    backtest = result.backtest_result
    print(f"Total Return: {backtest.metrics.total_return:.2%}")

    # Access derivatives stats
    print(f"Liquidations: {result.metrics['liquidations']}")
    print(f"Funding Paid: ${result.metrics['funding_paid']:.2f}")

    # View decision log
    for decision in backtest.decisions:
        print(f"  {decision.context.timestamp}: {decision.decision.action.value}")
```

### Combining with Strategy Optimization

```python
# Test multiple parameter sets
param_grid = [
    {"short_ma_period": 5, "long_ma_period": 20, "leverage": 5},
    {"short_ma_period": 10, "long_ma_period": 30, "leverage": 3},
    {"short_ma_period": 20, "long_ma_period": 50, "leverage": 2},
]

best_params = None
best_score = -float("inf")

for params in param_grid:
    # Configure strategy
    config = StrategyConfig(**params)
    strategy = BaselinePerpsStrategy(config=config)

    # Run stress tests
    validator = StressTestValidator(strategy=strategy, criteria=criteria)
    validator.run_standard_suite(data=data, symbol="BTC-PERP")

    # Score: +1 for pass, -10 for failure
    if validator.passed_all():
        score = 1.0
        # Bonus for better Sharpe
        avg_sharpe = sum(r.metrics["sharpe_ratio"] for r in validator.results) / len(validator.results)
        score += avg_sharpe

        if score > best_score:
            best_score = score
            best_params = params

print(f"Best parameters: {best_params}")
```

---

## Troubleshooting

### Failure: Liquidations Detected

**Symptoms**: `liquidation_count > 0`

**Root Causes**:
1. Leverage too high
2. Position size too large
3. Stops too wide
4. Not enough margin buffer

**Fixes**:
```python
# Reduce leverage
config.target_leverage = 3  # was 5

# Reduce position size
config.position_fraction = 0.10  # was 0.20

# Tighten stops
config.trailing_stop_pct = 0.015  # was 0.02

# Increase liquidation buffer
config.min_liquidation_buffer_pct = 0.20  # was 0.15
```

### Failure: Excessive Drawdown

**Symptoms**: `max_drawdown > max_drawdown_pct`

**Root Causes**:
1. Strategy holds losers too long
2. No stop losses
3. Adds to losing positions
4. Over-leveraged

**Fixes**:
```python
# Enable stops
config.trailing_stop_pct = 0.02  # 2% trailing stop

# Reduce leverage
config.target_leverage = 2

# Disable pyramiding
config.max_adds = 0  # Don't add to positions
```

### Failure: High Leverage Spike

**Symptoms**: `max_leverage > max_leverage` in criteria

**Root Causes**:
1. Position sizing calculation error
2. Multiple positions open simultaneously
3. Leverage not properly set

**Fixes**:
```python
# Ensure leverage is set correctly
config.target_leverage = 5

# Limit position count
config.max_positions = 1  # Only 1 position at a time

# Use conservative position sizing
config.position_fraction = 0.10
```

---

## Integration with Phase 2 Workflow

### Complete Phase 2 Checklist

```
[ ] 1. Run stress tests and ensure all pass
    ├── Gap moves (up and down)
    ├── High volatility
    ├── Flash crash
    ├── Funding shock
    └── Liquidity crisis

[ ] 2. Verify INTX eligibility
    ├── Check API key has INTX entitlements
    ├── Verify portfolio UUID resolution
    └── Test with fail-closed logic

[ ] 3. Strategy uplift (if desired)
    ├── Add volatility filter
    ├── Add trend strength filter
    └── Re-run stress tests with new strategy

[ ] 4. Run out-of-sample validation
    ├── Validate on different time periods
    ├── Validate across 3 regimes (trend/range/high-vol)
    └── Ensure beats MA-crossover baseline

[ ] 5. Go/No-Go Decision
    ├── All stress tests passed?
    ├── INTX eligible?
    ├── Strategy beats baseline?
    └── Ready for canary deployment
```

---

## See Also

- [Production-Parity Backtesting](../backtesting/production_parity_backtesting.md)
- [Derivatives Risk Management](./risk_management.md)
- [INTX Integration Guide](./intx_integration.md)
- [Liquidation Monitoring](./liquidation_monitoring.md)

---

## Appendix: Scenario Parameters Reference

### Default Scenario Settings

| Scenario | Parameter | Default | Range | Notes |
|----------|-----------|---------|-------|-------|
| Gap Move | `gap_size_pct` | 0.05 | 0.01-0.20 | 5% = moderate, 20% = extreme |
| Gap Move | `num_gaps` | 3 | 1-10 | More gaps = more stress |
| High Vol | `volatility_multiplier` | 2.0 | 1.5-5.0 | 2x = moderate, 5x = extreme |
| Flash Crash | `crash_size_pct` | 0.20 | 0.10-0.50 | 20% = severe, 50% = catastrophic |
| Flash Crash | `recovery_pct` | 0.50 | 0.0-1.0 | 50% = partial recovery |
| Funding Shock | `shock_rate` | 0.02 | 0.005-0.10 | 2% per 8h = very high |
| Liquidity Crisis | `slippage_multiplier` | 5.0 | 2.0-10.0 | 5x = severe illiquidity |

**Recommended**: Start with defaults, then increase severity if tests pass easily.
