# Backtest Feature Slice

## Purpose
Run historical backtests on trading strategies to evaluate performance.

## Entry Point
```python
from features.backtest import run_backtest

result = run_backtest(
    strategy="MomentumStrategy",
    symbol="AAPL", 
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    initial_capital=10000
)

print(result.summary())
```

## Files (Token Cost)

| File | Purpose | Lines | Tokens |
|------|---------|-------|--------|
| `backtest.py` | Main orchestration | 50 | ~50 |
| `data.py` | Data fetching and validation | 80 | ~80 |
| `signals.py` | Strategy signal generation | 60 | ~60 |
| `execution.py` | Trade simulation logic | 90 | ~90 |
| `metrics.py` | Performance metrics calculation | 70 | ~70 |
| `types.py` | Backtest-specific types | 40 | ~40 |

**Total Slice**: ~390 tokens

## Output
Returns `BacktestResult` containing:
- List of executed trades
- Portfolio equity curve
- Performance metrics (Sharpe, max drawdown, etc.)
- Trade statistics (win rate, profit factor, etc.)

## Dependencies
- `shared/strategies/` - Strategy implementations
- `shared/market_data.py` - Common data types

## Example Usage

### Simple Backtest
```python
result = run_backtest("SimpleMAStrategy", "AAPL", start, end)
print(f"Total Return: {result.metrics.total_return:.2f}%")
```

### With Custom Parameters
```python
result = run_backtest(
    strategy="MomentumStrategy",
    symbol="AAPL",
    start=start,
    end=end,
    initial_capital=50000,
    commission=0.001,
    slippage=0.0005
)
```

## Testing
```bash
python features/backtest/test_backtest.py
```