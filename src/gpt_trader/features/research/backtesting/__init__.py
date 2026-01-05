"""
Backtesting infrastructure for strategy evaluation.

This package provides:
- HistoricalDataLoader: Load market data from EventStore
- BacktestSimulator: Simulate strategy execution
- PerformanceMetrics: Calculate performance statistics
"""

from gpt_trader.features.research.backtesting.data_loader import (
    HistoricalDataLoader,
    HistoricalDataPoint,
)
from gpt_trader.features.research.backtesting.metrics import PerformanceMetrics
from gpt_trader.features.research.backtesting.simulator import (
    BacktestSimulator,
    SimulatedTrade,
)

__all__ = [
    "BacktestSimulator",
    "HistoricalDataLoader",
    "HistoricalDataPoint",
    "PerformanceMetrics",
    "SimulatedTrade",
]
