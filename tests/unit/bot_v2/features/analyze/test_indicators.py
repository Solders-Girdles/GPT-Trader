from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.analyze import indicators


def series(values: list[float]) -> pd.Series:
    return pd.Series(values, dtype=float)


def test_calculate_sma_and_ema_basic():
    data = series([1, 2, 3, 4, 5])

    sma = indicators.calculate_sma(data, period=3)
    ema = indicators.calculate_ema(data, period=3)

    assert np.isclose(sma.iloc[-1], (3 + 4 + 5) / 3)
    assert pytest.approx(ema.iloc[-1], rel=1e-6) == 4.0625


def test_calculate_rsi_handles_flat_series():
    flat = series([10, 10, 10, 10, 10])

    rsi = indicators.calculate_rsi(flat, period=3)

    assert (rsi == 0.0).all()


def test_calculate_rsi_with_price_moves():
    data = series([10, 11, 12, 11, 13, 12, 14])

    rsi = indicators.calculate_rsi(data, period=3)

    # Last value should be between 0 and 100
    assert 0 <= rsi.iloc[-1] <= 100


def test_calculate_macd_returns_components():
    data = series([i for i in range(1, 51)])

    macd_line, signal_line, histogram = indicators.calculate_macd(data)

    assert len(macd_line) == len(data)
    assert len(signal_line) == len(data)
    assert len(histogram) == len(data)
    assert np.isclose(histogram.iloc[-1], macd_line.iloc[-1] - signal_line.iloc[-1])


def test_calculate_bollinger_bands():
    data = series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    upper, middle, lower = indicators.calculate_bollinger_bands(data, period=5, num_std=2)

    assert len(upper) == len(data)
    assert np.isclose(middle.iloc[-1], data.tail(5).mean())
    assert np.isclose(lower.iloc[-1], middle.iloc[-1] - 2 * data.tail(5).std())


def test_calculate_atr_uses_true_range():
    high = series([10, 12, 13, 11, 14])
    low = series([8, 9, 10, 9, 11])
    close = series([9, 11, 12, 10, 13])

    atr = indicators.calculate_atr(high, low, close, period=3)

    assert len(atr) == len(high)
    assert atr.iloc[0] == 0.0
    assert atr.iloc[-1] >= 0


def test_calculate_obv_handles_price_moves():
    close = series([10, 11, 10, 12, 12])
    volume = series([100, 200, 150, 300, 250])

    obv = indicators.calculate_obv(close, volume)

    assert len(obv) == len(close)
    # OBV should increase on up moves, decrease on down moves, flat otherwise
    assert obv.iloc[1] > obv.iloc[0]
    assert obv.iloc[2] < obv.iloc[1]
    assert obv.iloc[4] == obv.iloc[3]


def test_calculate_obv_empty_close_returns_empty_series():
    close = pd.Series(dtype=float)
    volume = pd.Series(dtype=float)

    obv = indicators.calculate_obv(close, volume)

    assert obv.empty


def test_calculate_stochastic_returns_percent_k_and_d():
    high = series([10, 12, 13, 12, 14, 15, 16])
    low = series([8, 9, 10, 9, 11, 12, 13])
    close = series([9, 11, 12, 10, 13, 14, 15])

    percent_k, percent_d = indicators.calculate_stochastic(high, low, close, k_period=3, d_period=2)

    assert len(percent_k) == len(close)
    assert len(percent_d) == len(close)
    assert (percent_k >= 0).all()
    assert (percent_k <= 100).all()


def test_identify_support_resistance_handles_short_data():
    data = pd.DataFrame(
        {
            "high": [10, 11, 12],
            "low": [8, 7, 9],
            "close": [9, 10, 11],
        }
    )

    support1, support2, resistance1, resistance2, pivot = indicators.identify_support_resistance(
        data, lookback=5
    )

    assert support1 == 7
    assert support2 == pytest.approx((7 + 8 + 9) / 3)
    assert resistance1 == 12
    assert resistance2 == pytest.approx((12 + 11 + 10) / 3)
    assert pivot == pytest.approx((12 + 9 + 11) / 3)


def test_detect_trend_classifies_bullish_bearish_neutral():
    data = series([1, 2, 3, 4, 5, 6, 7])
    assert indicators.detect_trend(data, short_period=2, long_period=4) == "bullish"

    data_bear = series([7, 6, 5, 4, 3, 2, 1])
    assert indicators.detect_trend(data_bear, short_period=2, long_period=4) == "bearish"

    data_flat = series([5, 5, 5, 5, 5, 5, 5])
    assert indicators.detect_trend(data_flat, short_period=2, long_period=4) == "neutral"


@pytest.mark.parametrize(
    "returns,expected",
    [
        (series([0.001, -0.001] * 15), "low"),
        (series([0.012, -0.012] * 15), "medium"),
        (series([0.03, -0.03] * 15), "high"),
    ],
)
def test_calculate_volatility_classifies_regimes(returns: pd.Series, expected: str):
    result = indicators.calculate_volatility(returns, period=10)

    assert result == expected


def test_calculate_adx_returns_three_series():
    """Test that ADX returns ADX, +DI, and -DI series."""
    high = series([10, 12, 13, 14, 13, 15, 16, 15, 17, 18, 19, 18, 20, 21, 22])
    low = series([8, 9, 10, 11, 10, 12, 13, 12, 14, 15, 16, 15, 17, 18, 19])
    close = series([9, 11, 12, 13, 11, 14, 15, 13, 16, 17, 18, 16, 19, 20, 21])

    adx, plus_di, minus_di = indicators.calculate_adx(high, low, close, period=14)

    assert len(adx) == len(high)
    assert len(plus_di) == len(high)
    assert len(minus_di) == len(high)
    assert (adx >= 0).all()
    assert (adx <= 100).all()
    assert (plus_di >= 0).all()
    assert (minus_di >= 0).all()


def test_calculate_adx_trending_market():
    """Test ADX with a strong trending market."""
    # Create strong uptrend
    high = series([10 + i * 0.5 for i in range(30)])
    low = series([9 + i * 0.5 for i in range(30)])
    close = series([9.5 + i * 0.5 for i in range(30)])

    adx, plus_di, minus_di = indicators.calculate_adx(high, low, close, period=14)

    # In a strong uptrend, ADX should be high and +DI should exceed -DI
    assert adx.iloc[-1] > 20  # Strong trend
    assert plus_di.iloc[-1] > minus_di.iloc[-1]  # Uptrend


def test_calculate_adx_choppy_market():
    """Test ADX with a choppy/ranging market."""
    # Create sideways price action
    high = series([10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10])
    low = series([8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8])
    close = series([9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9])

    adx, plus_di, minus_di = indicators.calculate_adx(high, low, close, period=14)

    # In choppy markets, ADX should be relatively low
    assert adx.iloc[-1] < 40  # Weak trend or no trend


def test_calculate_adx_handles_flat_prices():
    """Test ADX with completely flat prices."""
    high = series([10] * 20)
    low = series([10] * 20)
    close = series([10] * 20)

    adx, plus_di, minus_di = indicators.calculate_adx(high, low, close, period=14)

    # All indicators should be zero or close to zero for flat prices
    assert adx.iloc[-1] < 1
    assert plus_di.iloc[-1] < 1
    assert minus_di.iloc[-1] < 1


def test_calculate_adx_different_periods():
    """Test ADX with different period settings."""
    high = series([10 + i * 0.3 for i in range(30)])
    low = series([9 + i * 0.3 for i in range(30)])
    close = series([9.5 + i * 0.3 for i in range(30)])

    adx_14, _, _ = indicators.calculate_adx(high, low, close, period=14)
    adx_7, _, _ = indicators.calculate_adx(high, low, close, period=7)

    # Shorter period should be more responsive
    assert len(adx_14) == len(high)
    assert len(adx_7) == len(high)
    # Both should detect the trend
    assert adx_14.iloc[-1] > 0
    assert adx_7.iloc[-1] > 0
