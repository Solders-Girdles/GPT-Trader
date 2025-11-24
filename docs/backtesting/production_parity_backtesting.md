# Production-Parity Backtesting

## Overview

The production-parity backtesting system ensures that backtests use the **exact same strategy code** as live trading, eliminating code duplication and guaranteeing decision alignment.

### Key Features

- ✅ **Zero Code Duplication**: Reuses `BaselinePerpsStrategy.decide()` directly
- ✅ **Decision Logging**: Records every decision with full context
- ✅ **Parity Validation**: Compare backtest vs live decision logs
- ✅ **Human-Readable Logs**: JSON format for easy inspection
- ✅ **Comprehensive Metrics**: Sharpe, drawdown, win rate, profit factor

---

## Quick Start

### Basic Usage

```python
from decimal import Decimal
import pandas as pd
from gpt_trader.features.live_trade.strategies.perps_baseline.config import StrategyConfig
from gpt_trader.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from gpt_trader.features.optimize.backtest_engine import run_backtest_production
from gpt_trader.features.optimize.types_v2 import BacktestConfig

# Load historical data
data = pd.read_csv("historical_btc.csv")  # Must have 'close' column

# Configure strategy (same config you use in production)
strategy_config = StrategyConfig(
    short_ma_period=5,
    long_ma_period=20,
    position_fraction=0.1,
    enable_shorts=False,
)

# Create strategy instance
strategy = BaselinePerpsStrategy(config=strategy_config)

# Configure backtest
backtest_config = BacktestConfig(
    initial_capital=Decimal("10000"),
    commission_rate=Decimal("0.001"),  # 0.1% = 10 bps
    slippage_rate=Decimal("0.0005"),   # 0.05% = 5 bps
    enable_decision_logging=True,
)

# Run backtest
result = run_backtest_production(
    strategy=strategy,
    data=data,
    symbol="BTC-USD",
    config=backtest_config,
)

# View results
print(result.summary())
```

---

## Decision Logging

Every strategy decision is logged with complete context:

### Log Structure

```json
{
  "run_id": "bt_20250122_143052_BTC-USD",
  "strategy_name": "BaselinePerpsStrategy",
  "symbol": "BTC-USD",
  "start_time": "2025-01-15T00:00:00",
  "end_time": "2025-01-31T23:59:59",
  "config": {
    "initial_capital": "10000",
    "commission_rate": "0.001",
    "slippage_rate": "0.0005"
  },
  "decisions": [
    {
      "context": {
        "timestamp": "2025-01-15T10:30:00",
        "symbol": "BTC-USD",
        "current_mark": "42350.50",
        "recent_marks": ["42320.10", "42340.25", ...],
        "position_state": null,
        "equity": "10500.00",
        "signal_label": "bullish"
      },
      "decision": {
        "action": "buy",
        "quantity": "0.1",
        "target_notional": "4235.05",
        "reason": "Bullish MA crossover"
      },
      "execution": {
        "filled": true,
        "fill_price": "42352.00",
        "filled_quantity": "0.1",
        "commission": "4.24",
        "slippage": "1.50"
      }
    }
  ],
  "metrics": {
    "total_return": 0.15,
    "sharpe_ratio": 1.2,
    "max_drawdown": 0.05,
    "win_rate": 0.6,
    "profit_factor": 1.5,
    "total_trades": 10
  }
}
```

### Log Storage

Logs are stored in: `backtesting/decision_logs/YYYY-MM-DD/bt_{timestamp}_{symbol}.json`

Example: `backtesting/decision_logs/2025-01-22/bt_20250122_143052_BTC-USD.json`

---

## Parity Validation

### Comparing Backtest vs Live

```python
from pathlib import Path
from gpt_trader.features.optimize.decision_logger import compare_decision_logs

# Compare decision logs
comparison = compare_decision_logs(
    backtest_log=Path("backtesting/decision_logs/2025-01-22/bt_20250122_143052_BTC-USD.json"),
    live_log=Path("live_trading/decision_logs/2025-01-22/live_143052_BTC-USD.json"),
)

print(f"Parity Rate: {comparison['parity_rate']:.2%}")
print(f"Mismatches: {comparison['mismatch_count']}/{comparison['backtest_decisions']}")

# View mismatch details
for mismatch in comparison['mismatches']:
    print(f"Timestamp: {mismatch['timestamp']}")
    print(f"  Backtest: {mismatch['backtest_action']} ({mismatch['backtest_reason']})")
    print(f"  Live: {mismatch['live_action']} ({mismatch['live_reason']})")
```

### Phase 1 Go/No-Go Criteria

**Success criteria**: Parity rate > 99% on 24-hour shadow run

- Same signals given identical candle streams
- Same guard violations
- Less than 1 trade difference per symbol

---

## Advanced Usage

### Custom Portfolio Simulation

```python
from gpt_trader.features.optimize.backtest_engine import BacktestEngine
from gpt_trader.features.optimize.types_v2 import BacktestConfig

# Create engine with custom config
config = BacktestConfig(
    initial_capital=Decimal("50000"),
    commission_rate=Decimal("0.002"),  # Higher fees
    slippage_rate=Decimal("0.001"),    # More slippage
    enable_decision_logging=True,
)

engine = BacktestEngine(strategy=strategy, config=config)
result = engine.run(data=data, symbol="BTC-USD")

# Access portfolio details
stats = engine.portfolio.get_stats()
print(f"Total Commission Paid: ${stats['total_commission']}")
print(f"Trade Count: {stats['trade_count']}")
```

### Loading and Analyzing Logs

```python
from gpt_trader.features.optimize.decision_logger import load_decision_log
from pathlib import Path

# Load a saved backtest
log_path = Path("backtesting/decision_logs/2025-01-22/bt_20250122_143052_BTC-USD.json")
result = load_decision_log(log_path)

# Analyze decisions
buy_decisions = [d for d in result.decisions if d.decision.action.value == "buy"]
sell_decisions = [d for d in result.decisions if d.decision.action.value == "sell"]

print(f"Buy signals: {len(buy_decisions)}")
print(f"Sell signals: {len(sell_decisions)}")

# Analyze execution quality
filled = [d for d in result.decisions if d.execution.filled]
avg_slippage = sum(float(d.execution.slippage) for d in filled) / len(filled)
print(f"Average slippage: ${avg_slippage:.2f}")
```

---

## Backtester Comparison

### Old Backtester (`run_backtest_local`)

- ❌ Uses separate `SimpleMAStrategy` implementation
- ❌ Different signal calculation (pandas rolling vs Decimal MA)
- ❌ No decision logging
- ❌ **NOT suitable for parity validation**
- ✅ Faster for parameter optimization

**Use case**: Rapid parameter sweeps during development

### New Backtester (`run_backtest_production`)

- ✅ Uses production `BaselinePerpsStrategy.decide()` exactly
- ✅ Same signal calculation as live trading
- ✅ Full decision logging with context
- ✅ **Required for Phase 1 parity validation**
- ⚠️ Slightly slower (more realistic)

**Use case**: Validating strategy before live deployment

---

## Integration with Live Trading

### Step 1: Run Backtest

```python
# Run backtest on recent data
result = run_backtest_production(
    strategy=strategy,
    data=last_24_hours_data,
    symbol="BTC-USD",
)
```

### Step 2: Deploy Live with Decision Logging

```python
# In your live trading loop, add decision logging:
from gpt_trader.features.optimize.decision_logger import DecisionLogger

logger = DecisionLogger(enabled=True, base_directory="live_trading/decision_logs")

# In each trading cycle:
decision = strategy.decide(
    symbol=symbol,
    current_mark=current_mark,
    position_state=position_state,
    recent_marks=recent_marks,
    equity=equity,
    product=product,
)

# Log decision with context
context = DecisionContext(
    timestamp=datetime.now(),
    symbol=symbol,
    current_mark=current_mark,
    recent_marks=list(recent_marks),
    position_state=position_state,
    equity=equity,
)

execution = execute_decision(decision)  # Your execution logic

logger.log_decision(context=context, decision=decision, execution=execution)
```

### Step 3: Validate Parity

```python
# Compare backtest vs live on same data
comparison = compare_decision_logs(
    backtest_log=backtest_path,
    live_log=live_path,
)

assert comparison['parity_rate'] > 0.99, "Parity validation failed!"
```

---

## Troubleshooting

### Issue: No trades in backtest

**Cause**: Insufficient data for MA calculation or no crossover signals

**Solution**:
- Ensure `len(data) > long_ma_period`
- Check MA periods in StrategyConfig
- Verify data contains volatility/trends

### Issue: Parity mismatch

**Possible causes**:
1. Different MA periods between backtest and live
2. Position state format differences
3. Equity calculation differences
4. Timestamps not aligned

**Debug**:
```python
# Inspect first mismatch
mismatches = comparison['mismatches']
if mismatches:
    first = mismatches[0]
    print(f"Backtest reason: {first['backtest_reason']}")
    print(f"Live reason: {first['live_reason']}")
```

### Issue: Decision log file not found

**Cause**: Logging disabled or different directory

**Solution**:
```python
config = BacktestConfig(enable_decision_logging=True)  # Ensure enabled
```

---

## Performance Considerations

### Memory Usage

- Each decision record: ~1-2 KB
- 1000 decisions = ~1-2 MB log file
- Consider disabling logging for long backtests if not needed

### Speed

- Production backtester: ~1000-5000 bars/second
- Old backtester: ~10000-50000 bars/second
- Trade-off: Accuracy vs speed

---

## Next Steps

1. **Phase 1 (Days 1-14)**: Use production backtester for parity validation
2. **Phase 2 (Weeks 3-8)**: Add derivatives stress testing
3. **Phase 3 (Months 3+)**: ML model backtesting with same framework

---

## API Reference

### `run_backtest_production()`

```python
def run_backtest_production(
    *,
    strategy: BaselinePerpsStrategy,
    data: pd.DataFrame,
    symbol: str,
    product: Product | None = None,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run production-parity backtest.

    Args:
        strategy: Production strategy instance
        data: Historical OHLC data (must have 'close' column)
        symbol: Trading symbol (e.g., "BTC-USD")
        product: Product metadata (auto-generated if None)
        config: Backtest configuration

    Returns:
        BacktestResult with decisions, metrics, equity curve
    """
```

### `BacktestConfig`

```python
@dataclass
class BacktestConfig:
    initial_capital: Decimal = Decimal("10000")
    commission_rate: Decimal = Decimal("0.001")   # 10 bps
    slippage_rate: Decimal = Decimal("0.0005")    # 5 bps
    enable_decision_logging: bool = True
    log_directory: str = "backtesting/decision_logs"
```

### `BacktestResult`

```python
@dataclass
class BacktestResult:
    run_id: str
    strategy_name: str
    symbol: str
    start_time: datetime
    end_time: datetime
    config: BacktestConfig
    decisions: list[DecisionRecord]
    metrics: BacktestMetrics
    equity_curve: list[tuple[datetime, Decimal]]

    def summary() -> str:
        """Human-readable summary"""

    def to_dict() -> dict:
        """JSON-serializable dict"""
```

---

## See Also

- [Strategy Configuration](../strategies/baseline_perps.md)
- [Decision Schemas](./decision_schemas.md)
- [Parity Validation Guide](./parity_validation.md)
