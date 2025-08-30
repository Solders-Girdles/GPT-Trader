"""
Backtest feature slice - self-contained backtesting functionality.
"""

from .backtest import run_backtest
from .types import BacktestResult, BacktestMetrics

__all__ = ['run_backtest', 'BacktestResult', 'BacktestMetrics']