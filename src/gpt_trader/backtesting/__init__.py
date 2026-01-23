"""
Backtesting simulation harness for GPT-Trader.

This module provides a production-grade backtesting framework that mirrors
live trading with swappable broker adapters, realistic order fills, and
comprehensive validation.

Key Components:
- SimulatedBroker: Broker implementation for backtesting
- ClockedBarRunner: Bar-by-bar historical data replay
- FeeCalculator: Coinbase Advanced Trade fee modeling
- OrderFillModel: Realistic fill simulation with slippage
- FundingPnLTracker: Perpetual futures funding tracking
- HistoricalDataManager: Data fetching and caching

Example:
    from gpt_trader.backtesting import SimulatedBroker, ClockedBarRunner
    from gpt_trader.backtesting.data import HistoricalDataManager
    from datetime import datetime

    # Setup
    broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))
    runner = ClockedBarRunner(
        data_provider=data_manager,
        symbols=["BTC-PERP-USDC", "ETH-PERP-USDC"],
        granularity="FIVE_MINUTE",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 3, 31),
    )

    # Run backtest
    async for bar_time, bars, quotes in runner.run():
        for symbol, bar in bars.items():
            broker.update_bar(symbol, bar)
        await strategy_coordinator.run_cycle()

    # Generate report
    report = broker.generate_report()
    print(report)
"""

from .chaos import ChaosEngine
from .engine import ClockedBarRunner, SimulationClock
from .metrics import BacktestReporter, RiskMetrics, TradeStatistics, generate_backtest_report
from .simulation import FeeCalculator, FundingPnLTracker, OrderFillModel, SimulatedBroker
from .types import (
    BacktestResult,
    ChaosScenario,
    ClockSpeed,
    FeeTier,
    SimulationConfig,
    ValidationDivergence,
    ValidationReport,
)
from .validation import DecisionLogger, GoldenPathValidator, StrategyDecision

__version__ = "1.0.0"

__all__ = [
    # Engine
    "ClockedBarRunner",
    "SimulationClock",
    # Simulation
    "SimulatedBroker",
    "OrderFillModel",
    "FeeCalculator",
    "FundingPnLTracker",
    # Metrics
    "TradeStatistics",
    "RiskMetrics",
    "BacktestReporter",
    "generate_backtest_report",
    # Validation
    "DecisionLogger",
    "StrategyDecision",
    "GoldenPathValidator",
    # Chaos Testing
    "ChaosEngine",
    # Types
    "SimulationConfig",
    "BacktestResult",
    "ValidationReport",
    "ValidationDivergence",
    "ChaosScenario",
    "ClockSpeed",
    "FeeTier",
]
