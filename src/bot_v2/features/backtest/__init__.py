"""
Backtest feature slice - self-contained backtesting functionality.

EXPERIMENTAL: This slice is provided for demos and local experimentation.
It is not part of the production perps trading path.
"""

from .backtest import run_backtest
from .profile import StrategySpec, build_strategy_spec, load_profile, run_profile_backtest
from .spot import (
    Bar,
    StrategySignal,
    SpotBacktestConfig,
    SpotBacktester,
    load_candles_from_parquet,
)
from .types import BacktestMetrics, BacktestResult

__all__ = [
    "run_backtest",
    "BacktestResult",
    "BacktestMetrics",
    "Bar",
    "StrategySignal",
    "SpotBacktestConfig",
    "SpotBacktester",
    "load_candles_from_parquet",
    "load_profile",
    "build_strategy_spec",
    "run_profile_backtest",
    "StrategySpec",
]

# Marker used by tooling and documentation
__experimental__ = True
