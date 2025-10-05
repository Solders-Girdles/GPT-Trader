# Optimize Feature

**Purpose**: Strategy backtesting, parameter optimization, and walk-forward analysis.

---

## Overview

The `optimize` feature provides:
- Historical backtesting engine
- Parameter optimization (grid search, random search)
- Walk-forward analysis
- Strategy performance comparison

**Coverage**: 🟢 98.9% (Excellent)

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.optimize import (
    Backtester,
    optimize_parameters,
    walk_forward_analysis
)
from bot_v2.shared.types import BacktestConfig, StrategyConfig
```

#### Data Requirements
- **Historical OHLCV**: pandas DataFrame with price/volume data
- **Strategy**: Callable strategy function or class
- **Parameter Space**: Dict defining optimization ranges

### Outputs

#### Data Structures
```python
from bot_v2.features.optimize.types import (
    BacktestResult,
    OptimizationResult,
    WalkForwardMetrics
)
```

#### Return Values
- **Backtest Result**: Trades, equity curve, performance metrics
- **Optimization Result**: Best parameters with performance scores
- **Walk-Forward Metrics**: Out-of-sample validation results

### Side Effects
- ✅ **Read-only**: No state modifications
- ✅ **Stateless**: Deterministic given same inputs
- 📊 **Metrics**: Emits backtest duration and iteration count

---

## Core Modules

### Backtester (`backtester.py`)
```python
class Backtester:
    """Historical simulation engine."""

    def run(
        self,
        strategy: Callable,
        data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000")
    ) -> BacktestResult:
        """Run backtest and return results."""
```

### Optimization (`optimize.py`)
```python
def optimize_parameters(
    strategy: Callable,
    data: pd.DataFrame,
    param_space: dict[str, list],
    method: str = "grid"  # grid, random, bayesian
) -> OptimizationResult:
    """Find optimal strategy parameters."""

def walk_forward_analysis(
    strategy: Callable,
    data: pd.DataFrame,
    train_size: int,
    test_size: int,
    step: int
) -> WalkForwardMetrics:
    """Validate strategy with walk-forward testing."""
```

---

## Usage Examples

### Basic Backtest
```python
from bot_v2.features.optimize import Backtester
from bot_v2.features.strategies import MomentumStrategy

backtester = Backtester()
strategy = MomentumStrategy(rsi_period=14)

result = backtester.run(
    strategy=strategy,
    data=historical_data,
    initial_capital=Decimal("10000")
)

print(f"Total Return: {result.performance.total_return:.2%}")
print(f"Sharpe Ratio: {result.performance.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.performance.max_drawdown:.2%}")
```

### Parameter Optimization
```python
from bot_v2.features.optimize import optimize_parameters

param_space = {
    "rsi_period": [10, 14, 20, 30],
    "rsi_oversold": [20, 25, 30],
    "rsi_overbought": [70, 75, 80]
}

result = optimize_parameters(
    strategy=MomentumStrategy,
    data=historical_data,
    param_space=param_space,
    method="grid"
)

print(f"Best params: {result.best_params}")
print(f"Best Sharpe: {result.best_score:.2f}")
```

### Walk-Forward Analysis
```python
from bot_v2.features.optimize import walk_forward_analysis

wf_result = walk_forward_analysis(
    strategy=MomentumStrategy,
    data=historical_data,
    train_size=252,  # 1 year training
    test_size=63,    # 3 months testing
    step=21          # Monthly reoptimization
)

print(f"In-sample Sharpe: {wf_result.in_sample_sharpe:.2f}")
print(f"Out-of-sample Sharpe: {wf_result.out_of_sample_sharpe:.2f}")
print(f"Degradation: {wf_result.degradation_pct:.2%}")
```

---

## Optimization Methods

### Grid Search
- **Pros**: Exhaustive, finds global optimum
- **Cons**: Slow for large parameter spaces (O(n^k) where k = parameters)
- **Use Case**: Small parameter spaces (< 1000 combinations)

### Random Search
- **Pros**: Fast, good for large spaces
- **Cons**: May miss optimal parameters
- **Use Case**: Large parameter spaces, time-constrained

### Bayesian Optimization (Future)
- **Pros**: Sample-efficient, finds optimum faster
- **Cons**: More complex implementation
- **Use Case**: Expensive objective functions

---

## Performance Metrics

All backtest results include:
- **Returns**: Total, annualized, daily
- **Risk**: Sharpe ratio, Sortino ratio, max drawdown
- **Trade Stats**: Win rate, profit factor, avg trade duration
- **Consistency**: Rolling Sharpe, monthly returns distribution

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/optimize/`)
- Backtest with synthetic price data
- Parameter optimization convergence
- Walk-forward window calculations

### Integration Tests
- Full backtest with real historical data
- Multi-year optimization runs
- Strategy comparison tests

---

## Configuration

```python
# Backtest settings
BACKTEST_INITIAL_CAPITAL = 10000
BACKTEST_COMMISSION_PCT = 0.001
BACKTEST_SLIPPAGE_BPS = 10

# Optimization settings
OPTIMIZE_MAX_ITERATIONS = 1000
OPTIMIZE_METRIC = "sharpe_ratio"  # sharpe_ratio, total_return, calmar_ratio
```

---

## Dependencies

### Internal
- `bot_v2.features.strategies` - Strategy implementations
- `bot_v2.shared.types` - Type definitions

### External
- `pandas` - Time series manipulation
- `numpy` - Numerical calculations

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
