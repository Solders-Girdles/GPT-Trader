# Optimization

Parameter optimization and strategy tuning utilities.

## Purpose

This package provides tools for optimizing strategy parameters through:

- **Walk-forward optimization**: Rolling window backtests with out-of-sample validation
- **Parameter sweeps**: Grid and random search over parameter spaces
- **Performance metrics**: Sharpe ratio, drawdown, win rate analysis

## Key Components

| Module | Purpose |
|--------|---------|
| `walk_forward.py` | Walk-forward optimizer with configurable windows |
| `trials.py` | Trial management and result aggregation |
| `metrics.py` | Performance calculation utilities |

## Usage

```python
from gpt_trader.features.optimize import WalkForwardOptimizer

optimizer = WalkForwardOptimizer(
    strategy_class=MyStrategy,
    param_space={"ma_period": [10, 20, 50]},
    train_window="30d",
    test_window="7d",
)

results = optimizer.run(historical_data)
```

## Output

Optimization results include:
- Best parameters per window
- Aggregate performance metrics
- Stability analysis across windows

## Related Packages

- `backtesting/` - Backtesting engine
- `features/data/` - Historical data fetching
