from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.analyze import strategies
from bot_v2.features.analyze.types import StrategySignals


def _dataframe_from_close(close: list[float]) -> pd.DataFrame:
    """Build minimal OHLC frame for tests."""
    close_series = pd.Series(close, dtype=float)
    return pd.DataFrame(
        {
            "close": close_series,
            "high": close_series + 1,
            "low": close_series - 1,
        }
    )


# Moving-average strategy ----------------------------------------------------


@pytest.mark.parametrize(
    "close,expected_signal,expected_reason",
    [
        ([3, 2, 1, 2, 3], 1, "Golden cross: 2MA crossed above 3MA"),
        ([1, 2, 3, 2, 1], -1, "Death cross: 2MA crossed below 3MA"),
        ([1, 1, 1, 1, 1], 0, "Bullish: 2MA above 3MA"),
        ([3, 3, 3, 3, 3], 0, "Bearish: 2MA below 3MA"),
        ([1, 2, 2, 2, 2], 0, "Bullish: 2MA above 3MA"),
        ([3, 2, 2, 2, 2], 0, "Bearish: 2MA below 3MA"),
    ],
)
def test_analyze_ma_strategy_detects_crossovers(
    close: list[int], expected_signal: int, expected_reason: str
) -> None:
    data = _dataframe_from_close(close)

    signal = strategies.analyze_ma_strategy(data, fast=2, slow=3)

    assert signal.signal == expected_signal
    assert signal.reason == expected_reason
    if expected_signal != 0:
        assert signal.confidence == pytest.approx(0.7)
    else:
        assert signal.confidence == pytest.approx(0.4)


def test_analyze_ma_strategy_with_insufficient_data() -> None:
    data = _dataframe_from_close([100, 101])

    signal = strategies.analyze_ma_strategy(data, fast=2, slow=3)

    assert signal == StrategySignals(
        strategy_name="Simple MA", signal=0, confidence=0.0, reason="Insufficient data"
    )


# Momentum strategy ---------------------------------------------------------


def test_analyze_momentum_strategy_positive_momentum() -> None:
    data = _dataframe_from_close([100, 102, 105, 108, 120])

    signal = strategies.analyze_momentum_strategy(data, lookback=3, threshold=0.02)

    assert signal.signal == 1
    assert signal.reason.startswith("Strong positive momentum")
    assert signal.confidence == pytest.approx(0.9)


def test_analyze_momentum_strategy_negative_momentum() -> None:
    data = _dataframe_from_close([120, 118, 115, 112, 100])

    signal = strategies.analyze_momentum_strategy(data, lookback=3, threshold=0.02)

    assert signal.signal == -1
    assert signal.reason.startswith("Strong negative momentum")
    assert signal.confidence == pytest.approx(0.9)


@pytest.mark.parametrize(
    "close,lookback,threshold,expected_signal,expected_confidence",
    [
        ([100, 101, 102, 102.5, 103], 3, 0.05, 0, 0.3),  # Neutral momentum
        ([100, 101, 102, 102.5, 103], 3, 0.01, 1, 0.5),  # Positive momentum with lower threshold
        ([103, 102.5, 102, 101, 100], 3, 0.01, -1, 0.5),  # Negative momentum with lower threshold
        ([100, 100, 100, 100, 100], 3, 0.01, 0, 0.3),  # No change
    ],
)
def test_analyze_momentum_strategy_various_conditions(
    close: list[float],
    lookback: int,
    threshold: float,
    expected_signal: int,
    expected_confidence: float,
) -> None:
    data = _dataframe_from_close(close)

    signal = strategies.analyze_momentum_strategy(data, lookback=lookback, threshold=threshold)

    assert signal.signal == expected_signal
    if expected_signal == 0:
        assert signal.reason.startswith("Neutral momentum")
    elif expected_signal == 1:
        assert signal.reason.startswith("Strong positive momentum")
    elif expected_signal == -1:
        assert signal.reason.startswith("Strong negative momentum")
    assert signal.confidence == pytest.approx(expected_confidence)


def test_analyze_momentum_strategy_with_insufficient_data() -> None:
    data = _dataframe_from_close([100, 101, 102])

    signal = strategies.analyze_momentum_strategy(data, lookback=3)

    assert signal == StrategySignals(
        strategy_name="Momentum", signal=0, confidence=0.0, reason="Insufficient data"
    )


# Mean reversion strategy ----------------------------------------------------


def test_analyze_mean_reversion_strategy_detects_oversold() -> None:
    data = _dataframe_from_close([10, 10, 10, 10, 8.8])

    signal = strategies.analyze_mean_reversion_strategy(data, period=5, num_std=1.5)

    assert signal.signal == 1
    assert signal.reason.startswith("Price at lower Bollinger Band")
    assert signal.confidence == pytest.approx(0.8)


def test_analyze_mean_reversion_strategy_detects_overbought() -> None:
    data = _dataframe_from_close([10, 10, 10, 10, 11.2])

    signal = strategies.analyze_mean_reversion_strategy(data, period=5, num_std=1.5)

    assert signal.signal == -1
    assert signal.reason.startswith("Price at upper Bollinger Band")
    assert signal.confidence == pytest.approx(0.8)


@pytest.mark.parametrize(
    "close,period,num_std,expected_signal,expected_confidence",
    [
        ([10, 10, 10, 10, 10.1], 5, 3.0, 0, 0.4),  # Within bands
        ([10, 10, 10, 10, 7.0], 5, 1.0, 1, 0.8),  # At lower band
        ([10, 10, 10, 10, 13.0], 5, 1.0, -1, 0.8),  # At upper band
        ([10, 10, 10, 10, 9.0], 5, 2.0, 0, 0.4),  # Within bands with different std
    ],
)
def test_analyze_mean_reversion_strategy_various_positions(
    close: list[float],
    period: int,
    num_std: float,
    expected_signal: int,
    expected_confidence: float,
) -> None:
    data = _dataframe_from_close(close)

    signal = strategies.analyze_mean_reversion_strategy(data, period=period, num_std=num_std)

    assert signal.signal == expected_signal
    if expected_signal == 0:
        assert signal.reason.startswith("Price within Bollinger Bands")
        assert signal.confidence == pytest.approx(expected_confidence)
    elif expected_signal == 1:
        assert signal.reason.startswith("Price at lower Bollinger Band")
        assert signal.confidence == pytest.approx(expected_confidence)
    elif expected_signal == -1:
        assert signal.reason.startswith("Price at upper Bollinger Band")
        assert signal.confidence == pytest.approx(expected_confidence)


def test_analyze_mean_reversion_strategy_with_insufficient_data() -> None:
    data = _dataframe_from_close([10, 10, 10])

    signal = strategies.analyze_mean_reversion_strategy(data, period=5)

    assert signal == StrategySignals(
        strategy_name="Mean Reversion", signal=0, confidence=0.0, reason="Insufficient data"
    )


# Volatility strategy --------------------------------------------------------


def test_analyze_volatility_strategy_high_volatility_flattens_signal() -> None:
    data = _dataframe_from_close([100, 120, 90, 130, 95])

    signal = strategies.analyze_volatility_strategy(data, period=3, vol_threshold=0.05)

    assert signal.signal == 0
    assert signal.reason.startswith("High volatility")
    assert signal.confidence == pytest.approx(0.7)


def test_analyze_volatility_strategy_positive_momentum_in_low_vol() -> None:
    data = pd.DataFrame(
        {
            "close": [100, 100.5, 101, 101.5, 103],
            "high": [101, 101.5, 102, 102.5, 104],
            "low": [99, 99.5, 100, 100.5, 102],
        }
    )

    signal = strategies.analyze_volatility_strategy(data, period=3, vol_threshold=0.05)

    assert signal.signal == 1
    assert "positive momentum" in signal.reason
    assert signal.confidence == pytest.approx(0.6)


def test_analyze_volatility_strategy_negative_momentum_in_low_vol() -> None:
    data = pd.DataFrame(
        {
            "close": [103, 101.5, 101, 100.5, 99],
            "high": [104, 102.5, 102, 101.5, 100],
            "low": [102, 100.5, 100, 99.5, 98],
        }
    )

    signal = strategies.analyze_volatility_strategy(data, period=3, vol_threshold=0.05)

    assert signal.signal == -1
    assert "negative momentum" in signal.reason
    assert signal.confidence == pytest.approx(0.6)


@pytest.mark.parametrize(
    "close,period,vol_threshold,expected_signal,expected_reason_contains",
    [
        ([100, 120, 90, 130, 95], 3, 0.05, 0, "High volatility"),  # High volatility
        (
            [100, 100.5, 101, 101.5, 103],
            3,
            0.05,
            1,
            "positive momentum",
        ),  # Low vol positive momentum
        (
            [103, 101.5, 101, 100.5, 99],
            3,
            0.05,
            -1,
            "negative momentum",
        ),  # Low vol negative momentum
        (
            [100, 100.1, 100.2, 100.3, 100.4],
            3,
            0.05,
            0,
            "no clear direction",
        ),  # Low vol no direction
    ],
)
def test_analyze_volatility_strategy_various_conditions(
    close: list[float],
    period: int,
    vol_threshold: float,
    expected_signal: int,
    expected_reason_contains: str,
) -> None:
    if expected_signal in [1, -1]:
        data = pd.DataFrame(
            {
                "close": close,
                "high": [c + 0.5 for c in close],
                "low": [c - 0.5 for c in close],
            }
        )
    else:
        data = _dataframe_from_close(close)

    signal = strategies.analyze_volatility_strategy(
        data, period=period, vol_threshold=vol_threshold
    )

    assert signal.signal == expected_signal
    assert expected_reason_contains in signal.reason
    if expected_signal == 0 and "High volatility" in expected_reason_contains:
        assert signal.confidence == pytest.approx(0.7)
    elif expected_signal in [1, -1]:
        assert signal.confidence == pytest.approx(0.6)
    elif expected_signal == 0 and "no clear direction" in expected_reason_contains:
        assert signal.confidence == pytest.approx(0.3)


def test_analyze_volatility_strategy_with_insufficient_data() -> None:
    data = _dataframe_from_close([100, 101, 102])

    signal = strategies.analyze_volatility_strategy(data, period=3)

    assert signal == StrategySignals(
        strategy_name="Volatility", signal=0, confidence=0.0, reason="Insufficient data"
    )


# Breakout strategy ----------------------------------------------------------


def test_analyze_breakout_strategy_detects_upward_breakout() -> None:
    data = pd.DataFrame(
        {
            "close": [100, 102, 101, 103, 110],
            "high": [101, 103, 102, 104, 111],
            "low": [99, 100, 99, 101, 105],
        }
    )

    signal = strategies.analyze_breakout_strategy(data, lookback=3)

    assert signal.signal == 1
    assert signal.reason.startswith("Upward breakout above 3-day high")
    assert signal.confidence == pytest.approx(0.9)


def test_analyze_breakout_strategy_detects_downward_breakout() -> None:
    data = pd.DataFrame(
        {
            "close": [110, 108, 109, 107, 100],
            "high": [111, 109, 110, 108, 101],
            "low": [109, 107, 108, 106, 99],
        }
    )

    signal = strategies.analyze_breakout_strategy(data, lookback=3)

    assert signal.signal == -1
    assert signal.reason.startswith("Downward breakout below 3-day low")
    assert signal.confidence == pytest.approx(0.9)


@pytest.mark.parametrize(
    "close,high,low,lookback,expected_signal,expected_reason_contains",
    [
        (
            [100, 105, 104, 106, 105],
            [101, 106, 105, 107, 106],
            [99, 104, 103, 104, 104],
            3,
            0,
            "Within 3-day range",
        ),  # Within range
        (
            [100, 102, 101, 103, 110],
            [101, 103, 102, 104, 111],
            [99, 100, 99, 101, 105],
            3,
            1,
            "Upward breakout",
        ),  # Upward breakout
        (
            [110, 108, 109, 107, 100],
            [111, 109, 110, 108, 101],
            [109, 107, 108, 106, 99],
            3,
            -1,
            "Downward breakout",
        ),  # Downward breakout
        (
            [100, 102, 101, 103, 102],
            [101, 103, 102, 104, 103],
            [99, 100, 99, 101, 101],
            3,
            0,
            "Within 3-day range",
        ),  # No breakout
    ],
)
def test_analyze_breakout_strategy_various_conditions(
    close: list[float],
    high: list[float],
    low: list[float],
    lookback: int,
    expected_signal: int,
    expected_reason_contains: str,
) -> None:
    data = pd.DataFrame(
        {
            "close": close,
            "high": high,
            "low": low,
        }
    )

    signal = strategies.analyze_breakout_strategy(data, lookback=lookback)

    assert signal.signal == expected_signal
    assert expected_reason_contains in signal.reason
    if expected_signal == 0:
        assert signal.confidence == pytest.approx(0.3)
    else:
        assert signal.confidence >= 0.6


def test_analyze_breakout_strategy_with_insufficient_data() -> None:
    data = pd.DataFrame({"close": [100, 101], "high": [101, 102], "low": [99, 100]})

    signal = strategies.analyze_breakout_strategy(data, lookback=3)

    assert signal == StrategySignals(
        strategy_name="Breakout", signal=0, confidence=0.0, reason="Insufficient data"
    )


# Orchestration --------------------------------------------------------------


@pytest.mark.parametrize(
    "data_size,expected_signals",
    [
        (
            40,
            ["Simple MA", "Momentum", "Mean Reversion", "Volatility", "Breakout"],
        ),  # Sufficient data
        (
            5,
            ["Simple MA", "Momentum", "Mean Reversion", "Volatility", "Breakout"],
        ),  # Minimal data, some strategies may return neutral
    ],
)
def test_analyze_with_strategies_returns_all_signals(
    data_size: int, expected_signals: list[str]
) -> None:
    close = np.linspace(100, 120, data_size)
    data = pd.DataFrame(
        {
            "close": close,
            "high": close + 1,
            "low": close - 1,
        }
    )

    signals = strategies.analyze_with_strategies(data)

    assert [s.strategy_name for s in signals] == expected_signals
    # Verify all signals have required attributes
    for signal in signals:
        assert hasattr(signal, "strategy_name")
        assert hasattr(signal, "signal")
        assert hasattr(signal, "confidence")
        assert hasattr(signal, "reason")
        assert signal.signal in [-1, 0, 1]
        assert 0.0 <= signal.confidence <= 1.0
