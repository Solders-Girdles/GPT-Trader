"""
Research feature slice for strategy backtesting and evaluation.

This module provides tools for:
- Loading and replaying historical market data
- Simulating strategy execution with realistic fills
- Calculating performance metrics
- Comparing strategies across different market conditions

Note: The canonical backtesting engine is `gpt_trader.backtesting`. The
research backtesting module is legacy and will migrate onto the canonical
engine via an adapter.

Example:
    from gpt_trader.features.research.backtesting import (
        HistoricalDataLoader,
        BacktestSimulator,
        PerformanceMetrics,
    )

    # Load historical data
    loader = HistoricalDataLoader(event_store)
    data = loader.load_time_range("BTC-USD", start, end)

    # Run backtest
    simulator = BacktestSimulator(initial_equity=Decimal("10000"))
    results = simulator.run(strategy, data)

    # Analyze performance
    metrics = PerformanceMetrics.from_results(results)
    print(f"Sharpe: {metrics.sharpe_ratio:.2f}")
"""

from gpt_trader.features.research.backtesting import (
    BacktestSimulator,
    HistoricalDataLoader,
    HistoricalDataPoint,
    PerformanceMetrics,
    SimulatedTrade,
)

__all__ = [
    "BacktestSimulator",
    "HistoricalDataLoader",
    "HistoricalDataPoint",
    "PerformanceMetrics",
    "SimulatedTrade",
]
