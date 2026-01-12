"""Backtesting engine components."""

from .bar_runner import ClockedBarRunner, IHistoricalDataProvider
from .clock import SimulationClock
from .guarded_execution import (
    BacktestDecisionContext,
    BacktestExecutionContext,
    BacktestGuardedExecutor,
)

__all__ = [
    "BacktestDecisionContext",
    "BacktestExecutionContext",
    "BacktestGuardedExecutor",
    "ClockedBarRunner",
    "IHistoricalDataProvider",
    "SimulationClock",
]
