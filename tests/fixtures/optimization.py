"""
Optimization test helpers.

Provides deterministic OHLC data, temporary workspaces, and canned backtest
metrics to keep optimization tests focused on behavioural assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.optimize.types import BacktestMetrics


def _generate_series(
    start: datetime,
    periods: int,
    *,
    base_price: float,
    drift: float,
    volatility: float,
    seed: int,
) -> pd.Series:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=drift, scale=volatility, size=periods)
    prices = np.cumsum(steps) + base_price
    return pd.Series(prices, index=pd.date_range(start=start, periods=periods, freq="1min"))


def build_ohlc_frame(
    *,
    start: datetime | None = None,
    periods: int = 120,
    base_price: float = 100.0,
    trend: float = 0.05,
    volatility: float = 0.5,
    seed: int = 7,
    include_nan: bool = False,
) -> pd.DataFrame:
    """
    Generate a deterministic OHLCV frame for strategy tests.

    Args:
        start: Starting timestamp (defaults to now minus ``periods`` minutes)
        periods: Number of bars
        base_price: Starting price
        trend: Deterministic upward drift added to the random walk
        volatility: Random walk standard deviation
        seed: RNG seed for reproducibility
        include_nan: Inject NaNs into the tail of the close series
    """

    start = start or (datetime.utcnow() - timedelta(minutes=periods))
    close = _generate_series(
        start=start,
        periods=periods,
        base_price=base_price,
        drift=trend,
        volatility=volatility,
        seed=seed,
    )

    open_prices = close.shift(1, fill_value=close.iloc[0])
    high = pd.concat([close, open_prices], axis=1).max(axis=1) + 0.25
    low = pd.concat([close, open_prices], axis=1).min(axis=1) - 0.25
    volume = pd.Series(10_000 + np.arange(periods), index=close.index, dtype=float)

    if include_nan and periods > 5:
        close.iloc[-3:] = np.nan

    frame = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return frame


@pytest.fixture
def ohlc_data_factory() -> Callable[..., pd.DataFrame]:
    """
    Factory for building deterministic OHLC frames.

    Usage:
        df = ohlc_data_factory(trend=0.1, volatility=0.2)
    """

    def _factory(**overrides) -> pd.DataFrame:
        return build_ohlc_frame(**overrides)

    return _factory


@pytest.fixture
def seeded_ohlc_sets(ohlc_data_factory):
    """
    Provide a suite of representative OHLC datasets.

    Returns a dict with keys ``uptrend``, ``downtrend``, ``flat``, and ``na_tail``.
    """

    return {
        "uptrend": ohlc_data_factory(trend=0.15, volatility=0.3, seed=1),
        "downtrend": ohlc_data_factory(trend=-0.12, volatility=0.35, seed=2),
        "flat": ohlc_data_factory(trend=0.0, volatility=0.05, seed=3),
        "na_tail": ohlc_data_factory(trend=0.08, volatility=0.4, seed=4, include_nan=True),
    }


@dataclass
class MetricsBuilder:
    """Helper to construct BacktestMetrics with sensible defaults."""

    base_return: float = 0.12
    sharpe: float = 1.5
    drawdown: float = -0.08
    win_rate: float = 0.58
    trades: int = 42

    def build(self, **overrides) -> BacktestMetrics:
        data = {
            "total_return": self.base_return,
            "sharpe_ratio": self.sharpe,
            "max_drawdown": self.drawdown,
            "win_rate": self.win_rate,
            "profit_factor": 1.8,
            "total_trades": self.trades,
            "avg_trade": 0.0015,
            "best_trade": 0.045,
            "worst_trade": -0.03,
            "recovery_factor": 1.2,
            "calmar_ratio": 0.9,
        }
        data.update(overrides)
        return BacktestMetrics(**data)


@pytest.fixture
def backtest_metrics_factory() -> MetricsBuilder:
    """Fixture returning a BacktestMetrics builder."""
    return MetricsBuilder()


@pytest.fixture
def fake_backtest_runner(backtest_metrics_factory):
    """
    Factory for patching ``run_backtest_local`` in tests that only need canned metrics.
    """

    def _factory(metrics: BacktestMetrics | None = None):
        result = metrics or backtest_metrics_factory.build()

        def _runner(*_args, **_kwargs):
            return result

        return _runner

    return _factory


@pytest.fixture
def optimization_workspace(tmp_path_factory) -> Path:
    """Temporary directory for optimization artefacts."""
    return tmp_path_factory.mktemp("optimize")


__all__ = [
    "backtest_metrics_factory",
    "fake_backtest_runner",
    "ohlc_data_factory",
    "optimization_workspace",
    "seeded_ohlc_sets",
]
