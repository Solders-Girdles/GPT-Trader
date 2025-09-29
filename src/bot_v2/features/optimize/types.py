"""
Local types for strategy optimization.

Complete isolation - no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ParameterGrid:
    """Grid of parameters to test."""

    strategy: str
    parameters: dict[str, list[Any]]

    def get_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations."""
        import itertools

        keys = list(self.parameters.keys())
        values = [self.parameters[k] for k in keys]

        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo, strict=False)))

        return combinations

    def total_combinations(self) -> int:
        """Count total combinations."""
        count = 1
        for values in self.parameters.values():
            count *= len(values)
        return count


@dataclass
class BacktestMetrics:
    """Metrics from a single backtest."""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade: float
    best_trade: float
    worst_trade: float
    recovery_factor: float
    calmar_ratio: float


@dataclass
class OptimizationResult:
    """Result from strategy optimization."""

    strategy: str
    symbol: str
    period: tuple[datetime, datetime]
    best_params: dict[str, Any]
    best_metrics: BacktestMetrics
    all_results: list[dict]  # All parameter combinations tested
    optimization_time: float  # Seconds

    def summary(self) -> str:
        """Generate optimization summary."""
        return f"""
Optimization Result for {self.strategy}
========================================
Symbol: {self.symbol}
Period: {self.period[0].date()} to {self.period[1].date()}
Combinations Tested: {len(self.all_results)}
Optimization Time: {self.optimization_time:.1f} seconds

Best Parameters:
{self._format_params()}

Best Performance:
- Total Return: {self.best_metrics.total_return:.2%}
- Sharpe Ratio: {self.best_metrics.sharpe_ratio:.2f}
- Max Drawdown: {self.best_metrics.max_drawdown:.2%}
- Win Rate: {self.best_metrics.win_rate:.2%}
- Total Trades: {self.best_metrics.total_trades}
        """.strip()

    def _format_params(self) -> str:
        """Format parameters nicely."""
        lines = []
        for key, value in self.best_params.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


@dataclass
class WalkForwardWindow:
    """Single window in walk-forward analysis."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_params: dict[str, Any]
    train_metrics: BacktestMetrics
    test_metrics: BacktestMetrics

    def get_efficiency(self) -> float:
        """Calculate walk-forward efficiency."""
        if self.train_metrics.total_return != 0:
            return self.test_metrics.total_return / self.train_metrics.total_return
        return 0.0


@dataclass
class WalkForwardResult:
    """Result from walk-forward analysis."""

    strategy: str
    symbol: str
    windows: list[WalkForwardWindow]
    avg_efficiency: float
    consistency_score: float  # 0-1, how consistent across windows
    robustness_score: float  # 0-1, overall robustness

    def summary(self) -> str:
        """Generate walk-forward summary."""
        avg_train_return = sum(w.train_metrics.total_return for w in self.windows) / len(
            self.windows
        )
        avg_test_return = sum(w.test_metrics.total_return for w in self.windows) / len(self.windows)

        return f"""
Walk-Forward Analysis for {self.strategy}
==========================================
Symbol: {self.symbol}
Windows: {len(self.windows)}
Avg Efficiency: {self.avg_efficiency:.2f}
Consistency: {self.consistency_score:.2%}
Robustness: {self.robustness_score:.2%}

Performance Summary:
- Avg Train Return: {avg_train_return:.2%}
- Avg Test Return: {avg_test_return:.2%}
- Best Window: Window {self._best_window_idx() + 1}
- Worst Window: Window {self._worst_window_idx() + 1}
        """.strip()

    def _best_window_idx(self) -> int:
        """Find best performing window."""
        returns = [w.test_metrics.total_return for w in self.windows]
        return returns.index(max(returns))

    def _worst_window_idx(self) -> int:
        """Find worst performing window."""
        returns = [w.test_metrics.total_return for w in self.windows]
        return returns.index(min(returns))


@dataclass
class SensitivityAnalysis:
    """Parameter sensitivity analysis."""

    parameter: str
    values: list[Any]
    metrics: dict[Any, BacktestMetrics]
    sensitivity_score: float  # How sensitive strategy is to this parameter
    optimal_value: Any
    stable_range: tuple[Any, Any]  # Range where performance is stable
