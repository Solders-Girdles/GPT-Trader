"""
Backtest-specific types and data structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Dict] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    metrics: 'BacktestMetrics' = None
    
    def summary(self) -> str:
        """Get a summary of results."""
        if self.metrics:
            return (
                f"Total Return: {self.metrics.total_return:.2f}%\n"
                f"Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}\n"
                f"Max Drawdown: {self.metrics.max_drawdown:.2f}%\n"
                f"Win Rate: {self.metrics.win_rate:.2f}%\n"
                f"Total Trades: {self.metrics.total_trades}"
            )
        return "No results available"


@dataclass
class BacktestMetrics:
    """Performance metrics for backtest."""
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0