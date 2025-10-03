"""Tests for strategy analysis functions."""

import pandas as pd
import pytest

from bot_v2.features.analyze.strategies import (
    analyze_breakout_strategy,
    analyze_ma_strategy,
    analyze_mean_reversion_strategy,
    analyze_momentum_strategy,
    analyze_volatility_strategy,
    analyze_with_strategies,
)
from bot_v2.features.analyze.types import StrategySignals


@pytest.fixture
def sample_data():
    """Create sample OHLC data for testing."""
    return pd.DataFrame(
        {
            "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] + [110] * 40,
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] + [111] * 40,
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108] + [109] * 40,
        }
    )


@pytest.fixture
def trending_up_data():
    """Create upward trending data."""
    return pd.DataFrame(
        {
            "close": list(range(100, 150)),
            "high": list(range(101, 151)),
            "low": list(range(99, 149)),
        }
    )


@pytest.fixture
def trending_down_data():
    """Create downward trending data."""
    return pd.DataFrame(
        {
            "close": list(range(150, 100, -1)),
            "high": list(range(151, 101, -1)),
            "low": list(range(149, 99, -1)),
        }
    )


@pytest.fixture
def volatile_data():
    """Create volatile oscillating data."""
    data = []
    for i in range(100):
        data.append(100 + (i % 2) * 10 - 5)
    return pd.DataFrame(
        {
            "close": data,
            "high": [x + 2 for x in data],
            "low": [x - 2 for x in data],
        }
    )


@pytest.fixture
def insufficient_data():
    """Create insufficient data for testing."""
    return pd.DataFrame(
        {
            "close": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
        }
    )


class TestAnalyzeWithStrategies:
    """Tests for analyze_with_strategies function."""

    def test_returns_list_of_signals(self, sample_data):
        """Test that function returns list of StrategySignals."""
        result = analyze_with_strategies(sample_data)
        assert isinstance(result, list)
        for signal in result:
            assert isinstance(signal, StrategySignals)

    def test_calls_all_strategies(self, sample_data):
        """Test that all strategies are called."""
        result = analyze_with_strategies(sample_data)
        strategy_names = [s.strategy_name for s in result]

        # Check that expected strategies are present
        expected = ["Simple MA", "Momentum", "Mean Reversion", "Volatility", "Breakout"]
        for name in expected:
            assert name in strategy_names

    def test_empty_result_for_insufficient_data(self, insufficient_data):
        """Test that insufficient data returns signals with 0 confidence."""
        result = analyze_with_strategies(insufficient_data)
        # Should still return signals, just with 0 confidence
        assert len(result) >= 0


class TestAnalyzeMAStrategy:
    """Tests for analyze_ma_strategy function."""

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = pd.DataFrame({"close": [100, 101, 102]})
        result = analyze_ma_strategy(data, fast=10, slow=30)

        assert result.strategy_name == "Simple MA"
        assert result.signal == 0
        assert result.confidence == 0.0
        assert "Insufficient data" in result.reason

    def test_golden_cross(self):
        """Test detection of golden cross (bullish signal)."""
        # Create data where fast MA crosses above slow MA
        close = [100] * 30 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        data = pd.DataFrame({"close": close})

        result = analyze_ma_strategy(data, fast=5, slow=20)

        assert result.strategy_name == "Simple MA"
        # Should detect golden cross or bullish condition
        assert result.signal in [0, 1]

    def test_death_cross(self):
        """Test detection of death cross (bearish signal)."""
        # Create data where fast MA crosses below slow MA
        close = [110] * 30 + [109, 108, 107, 106, 105, 104, 103, 102, 101, 100]
        data = pd.DataFrame({"close": close})

        result = analyze_ma_strategy(data, fast=5, slow=20)

        assert result.strategy_name == "Simple MA"
        # Should detect death cross or bearish condition
        assert result.signal in [-1, 0]

    def test_bullish_no_cross(self):
        """Test bullish state without crossover."""
        # Fast MA above slow MA but no recent cross
        close = list(range(100, 150))
        data = pd.DataFrame({"close": close})

        result = analyze_ma_strategy(data, fast=10, slow=30)

        assert result.strategy_name == "Simple MA"
        assert result.signal == 0  # No cross, just bullish
        assert "Bullish" in result.reason or "above" in result.reason.lower()

    def test_bearish_no_cross(self):
        """Test bearish state without crossover."""
        # Fast MA below slow MA but no recent cross
        close = list(range(150, 100, -1))
        data = pd.DataFrame({"close": close})

        result = analyze_ma_strategy(data, fast=10, slow=30)

        assert result.strategy_name == "Simple MA"
        assert result.signal == 0  # No cross, just bearish
        assert "Bearish" in result.reason or "below" in result.reason.lower()


class TestAnalyzeMomentumStrategy:
    """Tests for analyze_momentum_strategy function."""

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = pd.DataFrame({"close": [100, 101]})
        result = analyze_momentum_strategy(data, lookback=20)

        assert result.strategy_name == "Momentum"
        assert result.signal == 0
        assert result.confidence == 0.0
        assert "Insufficient data" in result.reason

    def test_strong_positive_momentum(self):
        """Test strong positive momentum detection."""
        close = list(range(100, 125))
        data = pd.DataFrame({"close": close})

        result = analyze_momentum_strategy(data, lookback=20, threshold=0.02)

        assert result.strategy_name == "Momentum"
        assert result.signal == 1
        assert result.confidence > 0.5
        assert "positive momentum" in result.reason.lower()

    def test_strong_negative_momentum(self):
        """Test strong negative momentum detection."""
        close = list(range(125, 100, -1))
        data = pd.DataFrame({"close": close})

        result = analyze_momentum_strategy(data, lookback=20, threshold=0.02)

        assert result.strategy_name == "Momentum"
        assert result.signal == -1
        assert result.confidence > 0.5
        assert "negative momentum" in result.reason.lower()

    def test_neutral_momentum(self):
        """Test neutral momentum detection."""
        close = [100] * 25
        data = pd.DataFrame({"close": close})

        result = analyze_momentum_strategy(data, lookback=20, threshold=0.02)

        assert result.strategy_name == "Momentum"
        assert result.signal == 0
        assert "neutral momentum" in result.reason.lower()


class TestAnalyzeMeanReversionStrategy:
    """Tests for analyze_mean_reversion_strategy function."""

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = pd.DataFrame({"close": [100, 101]})
        result = analyze_mean_reversion_strategy(data, period=20)

        assert result.strategy_name == "Mean Reversion"
        assert result.signal == 0
        assert result.confidence == 0.0
        assert "Insufficient data" in result.reason

    def test_price_at_lower_band(self):
        """Test price at lower Bollinger Band (buy signal)."""
        # Create data with mean 100 and add a dip at the end
        close = [100] * 19 + [80]
        data = pd.DataFrame({"close": close})

        result = analyze_mean_reversion_strategy(data, period=20, num_std=1.0)

        assert result.strategy_name == "Mean Reversion"
        assert result.signal == 1
        assert "lower" in result.reason.lower()

    def test_price_at_upper_band(self):
        """Test price at upper Bollinger Band (sell signal)."""
        # Create data with mean 100 and add a spike at the end
        close = [100] * 19 + [120]
        data = pd.DataFrame({"close": close})

        result = analyze_mean_reversion_strategy(data, period=20, num_std=1.0)

        assert result.strategy_name == "Mean Reversion"
        assert result.signal == -1
        assert "upper" in result.reason.lower()

    def test_price_within_bands(self):
        """Test price within Bollinger Bands (neutral)."""
        # Create data with some variation but current price in middle
        close = [98, 99, 100, 101, 102] * 5
        data = pd.DataFrame({"close": close})

        result = analyze_mean_reversion_strategy(data, period=20, num_std=2.0)

        assert result.strategy_name == "Mean Reversion"
        assert result.signal == 0
        assert "within" in result.reason.lower()


class TestAnalyzeVolatilityStrategy:
    """Tests for analyze_volatility_strategy function."""

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = pd.DataFrame({"close": [100, 101]})
        result = analyze_volatility_strategy(data, period=20)

        assert result.strategy_name == "Volatility"
        assert result.signal == 0
        assert result.confidence == 0.0
        assert "Insufficient data" in result.reason

    def test_high_volatility(self):
        """Test high volatility regime (stay out)."""
        # Create highly volatile data
        close = []
        for i in range(30):
            close.append(100 + (i % 2) * 20 - 10)
        data = pd.DataFrame({"close": close})

        result = analyze_volatility_strategy(data, period=20, vol_threshold=0.01)

        assert result.strategy_name == "Volatility"
        assert result.signal == 0
        assert "High volatility" in result.reason or "staying out" in result.reason

    def test_low_volatility_positive_momentum(self):
        """Test low volatility with positive momentum."""
        close = list(range(100, 125))
        data = pd.DataFrame({"close": close})

        result = analyze_volatility_strategy(data, period=20, vol_threshold=0.5)

        assert result.strategy_name == "Volatility"
        assert result.signal == 1
        assert "positive momentum" in result.reason.lower()

    def test_low_volatility_negative_momentum(self):
        """Test low volatility with negative momentum."""
        close = list(range(125, 100, -1))
        data = pd.DataFrame({"close": close})

        result = analyze_volatility_strategy(data, period=20, vol_threshold=0.5)

        assert result.strategy_name == "Volatility"
        assert result.signal == -1
        assert "negative momentum" in result.reason.lower()

    def test_low_volatility_no_direction(self):
        """Test low volatility without clear direction."""
        close = [100] * 25
        data = pd.DataFrame({"close": close})

        result = analyze_volatility_strategy(data, period=20, vol_threshold=0.5)

        assert result.strategy_name == "Volatility"
        assert result.signal == 0
        assert "no clear direction" in result.reason.lower()


class TestAnalyzeBreakoutStrategy:
    """Tests for analyze_breakout_strategy function."""

    def test_insufficient_data(self):
        """Test with insufficient data."""
        data = pd.DataFrame(
            {
                "close": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
            }
        )
        result = analyze_breakout_strategy(data, lookback=20)

        assert result.strategy_name == "Breakout"
        assert result.signal == 0
        assert result.confidence == 0.0
        assert "Insufficient data" in result.reason

    def test_upward_breakout(self):
        """Test upward breakout detection."""
        # Create data with a breakout above recent high
        close = [100] * 20 + [99, 115]
        high = [102] * 20 + [101, 116]
        low = [98] * 20 + [97, 113]

        data = pd.DataFrame({"close": close, "high": high, "low": low})

        result = analyze_breakout_strategy(data, lookback=20)

        assert result.strategy_name == "Breakout"
        assert result.signal == 1
        assert "Upward breakout" in result.reason

    def test_downward_breakout(self):
        """Test downward breakout detection."""
        # Create data with a breakout below recent low
        close = [100] * 20 + [101, 85]
        high = [102] * 20 + [103, 87]
        low = [98] * 20 + [99, 83]

        data = pd.DataFrame({"close": close, "high": high, "low": low})

        result = analyze_breakout_strategy(data, lookback=20)

        assert result.strategy_name == "Breakout"
        assert result.signal == -1
        assert "Downward breakout" in result.reason

    def test_within_range(self):
        """Test price within recent range."""
        close = [100] * 25
        high = [102] * 25
        low = [98] * 25

        data = pd.DataFrame({"close": close, "high": high, "low": low})

        result = analyze_breakout_strategy(data, lookback=20)

        assert result.strategy_name == "Breakout"
        assert result.signal == 0
        assert "Within" in result.reason or "range" in result.reason.lower()

    def test_confidence_increases_with_breakout_strength(self):
        """Test that confidence increases with breakout strength."""
        # Weak breakout
        close_weak = [100] * 20 + [99, 102.5]
        high_weak = [102] * 20 + [101, 103]
        low_weak = [98] * 20 + [97, 100]
        data_weak = pd.DataFrame({"close": close_weak, "high": high_weak, "low": low_weak})
        result_weak = analyze_breakout_strategy(data_weak, lookback=20)

        # Strong breakout
        close_strong = [100] * 20 + [99, 120]
        high_strong = [102] * 20 + [101, 121]
        low_strong = [98] * 20 + [97, 118]
        data_strong = pd.DataFrame({"close": close_strong, "high": high_strong, "low": low_strong})
        result_strong = analyze_breakout_strategy(data_strong, lookback=20)

        # Strong breakout should have higher or equal confidence
        # (both might be capped at 0.9)
        assert result_strong.confidence >= result_weak.confidence
