"""
Backtest feature slice - self-contained backtesting functionality.

EXPERIMENTAL: This slice is provided for demos and local experimentation.
It is not part of the production perps trading path.
"""

from .backtest import run_backtest
from .types import BacktestResult, BacktestMetrics

__all__ = ['run_backtest', 'BacktestResult', 'BacktestMetrics']

# Marker used by tooling and documentation
__experimental__ = True
