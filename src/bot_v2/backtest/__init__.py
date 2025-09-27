"""Backtesting utilities for spot trading strategies."""

from .spot import (
    Bar,
    StrategySignal,
    BacktestMetrics,
    BacktestResult,
    SpotBacktestConfig,
    SpotBacktester,
    MovingAverageCrossStrategy,
    BollingerMeanReversionStrategy,
    VolatilityFilteredStrategy,
    VolumeConfirmationStrategy,
    MomentumOscillatorStrategy,
    TrendStrengthStrategy,
    load_candles_from_parquet,
)

__all__ = [
    "Bar",
    "StrategySignal",
    "BacktestMetrics",
    "BacktestResult",
    "SpotBacktestConfig",
    "SpotBacktester",
    "MovingAverageCrossStrategy",
    "BollingerMeanReversionStrategy",
    "VolatilityFilteredStrategy",
    "VolumeConfirmationStrategy",
    "MomentumOscillatorStrategy",
    "TrendStrengthStrategy",
    "load_candles_from_parquet",
]
