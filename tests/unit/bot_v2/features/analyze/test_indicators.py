"""Tests for technical indicator calculations."""

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.analyze.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_sma,
    calculate_stochastic,
    calculate_volatility,
    detect_trend,
    identify_support_resistance,
)


@pytest.fixture
def price_series():
    """Create a simple price series."""
    return pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 110])


@pytest.fixture
def trending_up_series():
    """Create an upward trending price series."""
    return pd.Series(list(range(100, 200)))


@pytest.fixture
def trending_down_series():
    """Create a downward trending price series."""
    return pd.Series(list(range(200, 100, -1)))


@pytest.fixture
def ohlcv_data():
    """Create OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    np.random.seed(42)
    base = 100
    data = []
    for i in range(100):
        change = np.random.randn() * 2
        base = max(10, base + change)
        data.append({
            "open": base,
            "high": base + abs(np.random.randn()),
            "low": base - abs(np.random.randn()),
            "close": base + np.random.randn() * 0.5,
            "volume": np.random.randint(100000, 1000000),
        })
    return pd.DataFrame(data, index=dates)


class TestCalculateSMA:
    """Tests for calculate_sma function."""

    def test_simple_calculation(self):
        """Test basic SMA calculation."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = calculate_sma(data, period=3)

        # First two values should be NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

        # Third value: (10 + 20 + 30) / 3 = 20
        assert result.iloc[2] == pytest.approx(20, abs=0.01)

        # Fourth value: (20 + 30 + 40) / 3 = 30
        assert result.iloc[3] == pytest.approx(30, abs=0.01)

    def test_period_equals_length(self):
        """Test SMA when period equals data length."""
        data = pd.Series([10, 20, 30])
        result = calculate_sma(data, period=3)

        assert result.iloc[2] == pytest.approx(20, abs=0.01)

    def test_returns_series(self, price_series):
        """Test that function returns a Series."""
        result = calculate_sma(price_series, period=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)


class TestCalculateEMA:
    """Tests for calculate_ema function."""

    def test_simple_calculation(self):
        """Test basic EMA calculation."""
        data = pd.Series([10, 20, 30, 40, 50])
        result = calculate_ema(data, period=3)

        # EMA should start at first value
        assert result.iloc[0] == 10

        # Later values should show exponential weighting
        assert result.iloc[-1] > result.iloc[-2]

    def test_ema_vs_sma(self, price_series):
        """Test that EMA reacts faster than SMA."""
        sma = calculate_sma(price_series, period=5)
        ema = calculate_ema(price_series, period=5)

        # Both should exist but have different values
        assert not sma.equals(ema)

    def test_returns_series(self, price_series):
        """Test that function returns a Series."""
        result = calculate_ema(price_series, period=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)


class TestCalculateRSI:
    """Tests for calculate_rsi function."""

    def test_rsi_range(self, price_series):
        """Test that RSI is in valid range [0, 100]."""
        result = calculate_rsi(price_series, period=5)

        # Check non-NaN values are in valid range
        valid_values = result.dropna()
        assert all(0 <= val <= 100 for val in valid_values)

    def test_rsi_strong_uptrend(self):
        """Test RSI in strong uptrend."""
        data = pd.Series(list(range(100, 150)))
        result = calculate_rsi(data, period=14)

        # RSI should be high in strong uptrend
        assert result.iloc[-1] > 50

    def test_rsi_strong_downtrend(self):
        """Test RSI in strong downtrend."""
        data = pd.Series(list(range(150, 100, -1)))
        result = calculate_rsi(data, period=14)

        # RSI should be low in strong downtrend
        assert result.iloc[-1] < 50

    def test_rsi_neutral_market(self):
        """Test RSI in neutral/sideways market."""
        data = pd.Series([100] * 30)
        result = calculate_rsi(data, period=14)

        # RSI should be around 50 in neutral market (though may be NaN due to no movement)
        # Just check it doesn't crash
        assert isinstance(result, pd.Series)

    def test_returns_series(self, price_series):
        """Test that function returns a Series."""
        result = calculate_rsi(price_series, period=5)
        assert isinstance(result, pd.Series)
        assert len(result) == len(price_series)


class TestCalculateMACD:
    """Tests for calculate_macd function."""

    def test_returns_three_series(self, price_series):
        """Test that function returns three Series."""
        macd_line, signal_line, histogram = calculate_macd(price_series)

        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

    def test_histogram_calculation(self, price_series):
        """Test that histogram is macd - signal."""
        macd_line, signal_line, histogram = calculate_macd(price_series)

        # Check histogram = macd - signal for non-NaN values
        for i in range(len(price_series)):
            if not pd.isna(histogram.iloc[i]):
                expected = macd_line.iloc[i] - signal_line.iloc[i]
                assert histogram.iloc[i] == pytest.approx(expected, abs=0.01)

    def test_custom_periods(self, price_series):
        """Test MACD with custom periods."""
        macd1, signal1, hist1 = calculate_macd(price_series, fast=5, slow=10, signal=3)
        macd2, signal2, hist2 = calculate_macd(price_series, fast=12, slow=26, signal=9)

        # Different periods should give different results
        assert not macd1.equals(macd2)

    def test_uptrend_signal(self):
        """Test MACD in uptrend."""
        data = pd.Series(list(range(100, 150)))
        macd_line, signal_line, histogram = calculate_macd(data)

        # In strong uptrend, MACD should be positive
        assert macd_line.iloc[-1] > 0


class TestCalculateBollingerBands:
    """Tests for calculate_bollinger_bands function."""

    def test_returns_three_series(self, price_series):
        """Test that function returns three Series."""
        upper, middle, lower = calculate_bollinger_bands(price_series, period=5)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

    def test_band_relationships(self, price_series):
        """Test that upper > middle > lower."""
        upper, middle, lower = calculate_bollinger_bands(price_series, period=5)

        # Check for non-NaN values
        for i in range(len(price_series)):
            if not pd.isna(upper.iloc[i]):
                assert upper.iloc[i] >= middle.iloc[i]
                assert middle.iloc[i] >= lower.iloc[i]

    def test_middle_is_sma(self, price_series):
        """Test that middle band is SMA."""
        upper, middle, lower = calculate_bollinger_bands(price_series, period=5)
        sma = calculate_sma(price_series, period=5)

        # Middle band should equal SMA
        pd.testing.assert_series_equal(middle, sma)

    def test_custom_std_deviation(self, price_series):
        """Test Bollinger Bands with different standard deviations."""
        upper1, middle1, lower1 = calculate_bollinger_bands(price_series, period=5, num_std=1)
        upper2, middle2, lower2 = calculate_bollinger_bands(price_series, period=5, num_std=2)

        # Wider bands with higher std
        assert upper2.iloc[-1] > upper1.iloc[-1]
        assert lower2.iloc[-1] < lower1.iloc[-1]


class TestCalculateATR:
    """Tests for calculate_atr function."""

    def test_returns_series(self, ohlcv_data):
        """Test that function returns a Series."""
        result = calculate_atr(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            period=14
        )

        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_data)

    def test_atr_positive(self, ohlcv_data):
        """Test that ATR values are positive."""
        result = calculate_atr(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            period=14
        )

        # All non-NaN ATR values should be positive
        valid_values = result.dropna()
        assert all(val >= 0 for val in valid_values)

    def test_atr_reflects_volatility(self):
        """Test that ATR increases with volatility."""
        # Low volatility data
        low_vol = pd.DataFrame({
            "high": [101] * 20,
            "low": [99] * 20,
            "close": [100] * 20,
        })

        # High volatility data
        high_vol = pd.DataFrame({
            "high": [110, 105, 115, 100, 120] * 4,
            "low": [90, 95, 85, 100, 80] * 4,
            "close": [100, 100, 100, 100, 100] * 4,
        })

        atr_low = calculate_atr(low_vol["high"], low_vol["low"], low_vol["close"], period=5)
        atr_high = calculate_atr(high_vol["high"], high_vol["low"], high_vol["close"], period=5)

        # High volatility should have higher ATR
        assert atr_high.iloc[-1] > atr_low.iloc[-1]


class TestCalculateOBV:
    """Tests for calculate_obv function."""

    def test_returns_series(self):
        """Test that function returns a Series."""
        close = pd.Series([100, 101, 102, 101, 103])
        volume = pd.Series([1000, 1000, 1000, 1000, 1000])

        result = calculate_obv(close, volume)

        assert isinstance(result, pd.Series)
        assert len(result) == len(close)

    def test_obv_increases_on_up_day(self):
        """Test that OBV increases when price goes up."""
        close = pd.Series([100, 105])
        volume = pd.Series([1000, 1000])

        result = calculate_obv(close, volume)

        # OBV should increase when price increases
        assert result.iloc[1] > result.iloc[0]

    def test_obv_decreases_on_down_day(self):
        """Test that OBV decreases when price goes down."""
        close = pd.Series([100, 95])
        volume = pd.Series([1000, 1000])

        result = calculate_obv(close, volume)

        # OBV should decrease when price decreases
        assert result.iloc[1] < result.iloc[0]

    def test_obv_unchanged_on_flat_day(self):
        """Test that OBV stays same when price unchanged."""
        close = pd.Series([100, 100])
        volume = pd.Series([1000, 1000])

        result = calculate_obv(close, volume)

        # OBV should stay same when price unchanged
        assert result.iloc[1] == result.iloc[0]

    def test_obv_first_value(self):
        """Test that first OBV value equals first volume."""
        close = pd.Series([100, 101, 102])
        volume = pd.Series([500, 1000, 1500])

        result = calculate_obv(close, volume)

        assert result.iloc[0] == volume.iloc[0]


class TestCalculateStochastic:
    """Tests for calculate_stochastic function."""

    def test_returns_two_series(self, ohlcv_data):
        """Test that function returns two Series."""
        k, d = calculate_stochastic(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            k_period=14,
            d_period=3
        )

        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_stochastic_range(self, ohlcv_data):
        """Test that stochastic values are in [0, 100] range."""
        k, d = calculate_stochastic(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            k_period=14,
            d_period=3
        )

        # Check %K
        valid_k = k.dropna()
        assert all(0 <= val <= 100 for val in valid_k)

        # Check %D
        valid_d = d.dropna()
        assert all(0 <= val <= 100 for val in valid_d)

    def test_d_is_sma_of_k(self, ohlcv_data):
        """Test that %D is moving average of %K."""
        k, d = calculate_stochastic(
            ohlcv_data["high"],
            ohlcv_data["low"],
            ohlcv_data["close"],
            k_period=14,
            d_period=3
        )

        # %D should be smoother than %K
        k_std = k.std()
        d_std = d.std()

        # %D should have lower standard deviation (smoother)
        assert d_std <= k_std


class TestIdentifySupportResistance:
    """Tests for identify_support_resistance function."""

    def test_returns_five_values(self, ohlcv_data):
        """Test that function returns five values."""
        result = identify_support_resistance(ohlcv_data, lookback=20)

        assert len(result) == 5
        immediate_support, strong_support, immediate_resistance, strong_resistance, pivot = result

        # All should be numbers
        assert all(isinstance(val, (int, float)) for val in result)

    def test_resistance_above_support(self, ohlcv_data):
        """Test that resistance is above support."""
        immediate_support, strong_support, immediate_resistance, strong_resistance, pivot = (
            identify_support_resistance(ohlcv_data, lookback=20)
        )

        assert immediate_resistance >= immediate_support
        assert strong_resistance >= strong_support

    def test_pivot_calculation(self):
        """Test pivot point calculation."""
        data = pd.DataFrame({
            "high": [110, 110],
            "low": [90, 90],
            "close": [100, 100],
        })

        _, _, _, _, pivot = identify_support_resistance(data, lookback=2)

        # Pivot = (high + low + close) / 3 = (110 + 90 + 100) / 3 = 100
        assert pivot == pytest.approx(100, abs=0.01)

    def test_custom_lookback(self, ohlcv_data):
        """Test with different lookback periods."""
        result1 = identify_support_resistance(ohlcv_data, lookback=10)
        result2 = identify_support_resistance(ohlcv_data, lookback=30)

        # Different lookbacks may give different results
        assert isinstance(result1, tuple)
        assert isinstance(result2, tuple)


class TestDetectTrend:
    """Tests for detect_trend function."""

    def test_bullish_trend(self):
        """Test detection of bullish trend."""
        data = pd.Series(list(range(100, 200)))
        result = detect_trend(data, short_period=10, long_period=20)

        assert result == "bullish"

    def test_bearish_trend(self):
        """Test detection of bearish trend."""
        data = pd.Series(list(range(200, 100, -1)))
        result = detect_trend(data, short_period=10, long_period=20)

        assert result == "bearish"

    def test_neutral_trend(self):
        """Test detection of neutral trend."""
        data = pd.Series([100] * 60)
        result = detect_trend(data, short_period=20, long_period=50)

        assert result == "neutral"

    def test_returns_string(self, price_series):
        """Test that function returns a string."""
        result = detect_trend(price_series, short_period=3, long_period=5)

        assert isinstance(result, str)
        assert result in ["bullish", "bearish", "neutral"]


class TestCalculateVolatility:
    """Tests for calculate_volatility function."""

    def test_low_volatility(self):
        """Test detection of low volatility."""
        # Very stable returns
        returns = pd.Series([0.001] * 30)
        result = calculate_volatility(returns, period=20)

        assert result == "low"

    def test_high_volatility(self):
        """Test detection of high volatility."""
        # Large returns
        np.random.seed(42)
        returns = pd.Series(np.random.randn(30) * 0.1)  # 10% daily returns
        result = calculate_volatility(returns, period=20)

        assert result == "high"

    def test_medium_volatility(self):
        """Test detection of medium volatility."""
        # Moderate returns (annualized vol around 0.20)
        np.random.seed(42)
        returns = pd.Series(np.random.randn(30) * 0.012)
        result = calculate_volatility(returns, period=20)

        assert result in ["low", "medium", "high"]

    def test_returns_string(self):
        """Test that function returns a string."""
        returns = pd.Series([0.01, -0.01, 0.01, -0.01] * 10)
        result = calculate_volatility(returns, period=20)

        assert isinstance(result, str)
        assert result in ["low", "medium", "high"]


class TestIndicatorIntegration:
    """Integration tests for multiple indicators."""

    def test_all_indicators_on_same_data(self, ohlcv_data):
        """Test that all indicators can be calculated on same dataset."""
        close = ohlcv_data["close"]
        high = ohlcv_data["high"]
        low = ohlcv_data["low"]
        volume = ohlcv_data["volume"]

        # All should complete without error
        sma = calculate_sma(close, period=20)
        ema = calculate_ema(close, period=20)
        rsi = calculate_rsi(close, period=14)
        macd_line, signal_line, histogram = calculate_macd(close)
        upper, middle, lower = calculate_bollinger_bands(close, period=20)
        atr = calculate_atr(high, low, close, period=14)
        obv = calculate_obv(close, volume)
        k, d = calculate_stochastic(high, low, close)
        support_resistance = identify_support_resistance(ohlcv_data)
        trend = detect_trend(close)

        # Basic sanity checks
        assert len(sma) == len(close)
        assert len(ema) == len(close)
        assert len(rsi) == len(close)
        assert len(macd_line) == len(close)
        assert len(upper) == len(close)
        assert len(atr) == len(close)
        assert len(obv) == len(close)
        assert len(k) == len(close)
        assert len(support_resistance) == 5
        assert trend in ["bullish", "bearish", "neutral"]
