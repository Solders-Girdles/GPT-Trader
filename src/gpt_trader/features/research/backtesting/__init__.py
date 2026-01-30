"""
Legacy backtesting infrastructure for strategy evaluation.

This package provides:
- HistoricalDataLoader: Load market data from EventStore
- BacktestSimulator: Simulate strategy execution
- PerformanceMetrics: Calculate performance statistics

Canonical backtesting lives in `gpt_trader.backtesting`; this module will be
replaced by an adapter in a future consolidation.
"""

from gpt_trader.features.research.backtesting.data_loader import (
    HistoricalDataLoader,
    HistoricalDataPoint,
)
from gpt_trader.features.research.backtesting.adapter import BacktestSimulator
from gpt_trader.features.research.backtesting.metrics import PerformanceMetrics
from gpt_trader.features.research.backtesting.simulator import SimulatedTrade

__all__ = [
    "BacktestSimulator",
    "HistoricalDataLoader",
    "HistoricalDataPoint",
    "PerformanceMetrics",
    "SimulatedTrade",
]
