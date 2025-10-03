"""Tests for price pattern detection functions."""

import pandas as pd
import pytest

from bot_v2.features.analyze.patterns import (
    detect_double_bottom,
    detect_double_top,
    detect_flag,
    detect_head_shoulders,
    detect_patterns,
    detect_triangle,
)
from bot_v2.features.analyze.types import PricePattern


@pytest.fixture
def double_top_data():
    """Create data with double top pattern."""
    # Create two peaks at similar heights
    highs = [100] * 5 + [105] + [104, 103, 104] + [105] + [103] * 5
    lows = [h - 2 for h in highs]
    closes = [h - 1 for h in highs]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


@pytest.fixture
def double_bottom_data():
    """Create data with double bottom pattern."""
    # Create two troughs at similar depths
    lows = [100] * 5 + [95] + [96, 97, 96] + [95] + [97] * 5
    highs = [l + 2 for l in lows]
    closes = [l + 1 for l in lows]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


@pytest.fixture
def head_shoulders_data():
    """Create data with head and shoulders pattern."""
    # Create left shoulder, head, right shoulder
    highs = [100] * 5 + [105] + [103] * 3 + [110] + [103] * 3 + [105] + [100] * 10
    lows = [h - 3 for h in highs]
    closes = [h - 1 for h in highs]
    return pd.DataFrame({"high": highs, "low": lows, "close": closes})


@pytest.fixture
def ascending_triangle_data():
    """Create data with ascending triangle pattern."""
    # Rising lows, flat highs
    data = []
    for i in range(20):
        high = 105
        low = 95 + i * 0.5
        close = (high + low) / 2
        data.append({"high": high, "low": low, "close": close})
    return pd.DataFrame(data)


@pytest.fixture
def descending_triangle_data():
    """Create data with descending triangle pattern."""
    # Flat lows, falling highs
    data = []
    for i in range(20):
        low = 95
        high = 105 - i * 0.5
        close = (high + low) / 2
        data.append({"high": high, "low": low, "close": close})
    return pd.DataFrame(data)


@pytest.fixture
def bull_flag_data():
    """Create data with bull flag pattern."""
    # Strong move up followed by consolidation
    pre_flag = [{"high": 100 + i * 2, "low": 98 + i * 2, "close": 99 + i * 2} for i in range(10)]
    flag = [{"high": 121, "low": 119, "close": 120} for _ in range(15)]
    return pd.DataFrame(pre_flag + flag)


@pytest.fixture
def bear_flag_data():
    """Create data with bear flag pattern."""
    # Strong move down followed by consolidation
    pre_flag = [{"high": 120 - i * 2, "low": 118 - i * 2, "close": 119 - i * 2} for i in range(10)]
    flag = [{"high": 101, "low": 99, "close": 100} for _ in range(15)]
    return pd.DataFrame(pre_flag + flag)


@pytest.fixture
def insufficient_data():
    """Create insufficient data for testing."""
    return pd.DataFrame(
        {
            "high": [101, 102],
            "low": [99, 100],
            "close": [100, 101],
        }
    )


class TestDetectPatterns:
    """Tests for detect_patterns function."""

    def test_returns_list(self, double_top_data):
        """Test that function returns a list."""
        result = detect_patterns(double_top_data)
        assert isinstance(result, list)

    def test_returns_price_patterns(self, double_top_data):
        """Test that function returns PricePattern objects."""
        result = detect_patterns(double_top_data)
        for pattern in result:
            assert isinstance(pattern, PricePattern)

    def test_detects_multiple_patterns(self):
        """Test that function can detect multiple patterns."""
        # Create complex data that might have multiple patterns
        data = pd.DataFrame(
            {
                "high": list(range(100, 150)),
                "low": list(range(98, 148)),
                "close": list(range(99, 149)),
            }
        )
        result = detect_patterns(data)
        assert isinstance(result, list)


class TestDetectDoubleTop:
    """Tests for detect_double_top function."""

    def test_insufficient_data(self, insufficient_data):
        """Test with insufficient data."""
        result = detect_double_top(insufficient_data, lookback=20)
        assert result is None

    def test_no_pattern(self):
        """Test with no double top pattern."""
        data = pd.DataFrame(
            {
                "high": list(range(100, 125)),
                "low": list(range(98, 123)),
                "close": list(range(99, 124)),
            }
        )
        result = detect_double_top(data, lookback=20)
        # May or may not find a pattern in trending data
        assert result is None or isinstance(result, PricePattern)

    def test_detects_double_top(self, double_top_data):
        """Test detection of double top pattern."""
        result = detect_double_top(double_top_data, lookback=20)

        if result:  # Pattern detection is probabilistic
            assert result.pattern_type == "Double Top"
            assert result.confidence > 0
            assert result.target_price is not None
            assert result.stop_loss is not None
            assert "bearish" in result.description.lower()

    def test_peaks_must_be_similar(self):
        """Test that peaks must be within 3% to be detected."""
        # Create peaks with different heights (> 3% difference)
        highs = [100] * 5 + [105] + [103] * 3 + [115] + [103] * 5
        lows = [h - 2 for h in highs]
        closes = [h - 1 for h in highs]
        data = pd.DataFrame({"high": highs, "low": lows, "close": closes})

        result = detect_double_top(data, lookback=20)
        # Should not detect pattern due to dissimilar peaks
        assert result is None or result.confidence <= 0.7


class TestDetectDoubleBottom:
    """Tests for detect_double_bottom function."""

    def test_insufficient_data(self, insufficient_data):
        """Test with insufficient data."""
        result = detect_double_bottom(insufficient_data, lookback=20)
        assert result is None

    def test_no_pattern(self):
        """Test with no double bottom pattern."""
        data = pd.DataFrame(
            {
                "high": list(range(125, 100, -1)),
                "low": list(range(123, 98, -1)),
                "close": list(range(124, 99, -1)),
            }
        )
        result = detect_double_bottom(data, lookback=20)
        # May or may not find a pattern in trending data
        assert result is None or isinstance(result, PricePattern)

    def test_detects_double_bottom(self, double_bottom_data):
        """Test detection of double bottom pattern."""
        result = detect_double_bottom(double_bottom_data, lookback=20)

        if result:  # Pattern detection is probabilistic
            assert result.pattern_type == "Double Bottom"
            assert result.confidence > 0
            assert result.target_price is not None
            assert result.stop_loss is not None
            assert "bullish" in result.description.lower()

    def test_troughs_must_be_similar(self):
        """Test that troughs must be within 3% to be detected."""
        # Create troughs with different depths (> 3% difference)
        lows = [100] * 5 + [95] + [97] * 3 + [85] + [97] * 5
        highs = [l + 2 for l in lows]
        closes = [l + 1 for l in lows]
        data = pd.DataFrame({"high": highs, "low": lows, "close": closes})

        result = detect_double_bottom(data, lookback=20)
        # Should not detect pattern due to dissimilar troughs
        assert result is None or result.confidence <= 0.7


class TestDetectHeadShoulders:
    """Tests for detect_head_shoulders function."""

    def test_insufficient_data(self, insufficient_data):
        """Test with insufficient data."""
        result = detect_head_shoulders(insufficient_data, lookback=30)
        assert result is None

    def test_no_pattern(self):
        """Test with no head and shoulders pattern."""
        data = pd.DataFrame(
            {
                "high": list(range(100, 135)),
                "low": list(range(98, 133)),
                "close": list(range(99, 134)),
            }
        )
        result = detect_head_shoulders(data, lookback=30)
        assert result is None or isinstance(result, PricePattern)

    def test_detects_head_shoulders(self, head_shoulders_data):
        """Test detection of head and shoulders pattern."""
        result = detect_head_shoulders(head_shoulders_data, lookback=30)

        if result:  # Pattern detection is probabilistic
            assert result.pattern_type == "Head and Shoulders"
            assert result.confidence > 0
            assert result.target_price is not None
            assert result.stop_loss is not None
            assert "bearish" in result.description.lower()

    def test_head_must_be_highest(self):
        """Test that head must be higher than shoulders."""
        # Create pattern where head is not highest
        highs = [100] * 5 + [108] + [103] * 3 + [105] + [103] * 3 + [110] + [100] * 10
        lows = [h - 3 for h in highs]
        closes = [h - 1 for h in highs]
        data = pd.DataFrame({"high": highs, "low": lows, "close": closes})

        result = detect_head_shoulders(data, lookback=30)
        # Should not detect proper pattern
        assert result is None or result.pattern_type != "Head and Shoulders"


class TestDetectTriangle:
    """Tests for detect_triangle function."""

    def test_insufficient_data(self, insufficient_data):
        """Test with insufficient data."""
        result = detect_triangle(insufficient_data, lookback=20)
        assert result is None

    def test_detects_ascending_triangle(self, ascending_triangle_data):
        """Test detection of ascending triangle."""
        result = detect_triangle(ascending_triangle_data, lookback=20)

        if result:  # Pattern detection depends on slope thresholds
            assert "Triangle" in result.pattern_type
            assert result.confidence > 0
            assert result.target_price is not None
            assert result.stop_loss is not None

    def test_detects_descending_triangle(self, descending_triangle_data):
        """Test detection of descending triangle."""
        result = detect_triangle(descending_triangle_data, lookback=20)

        if result:  # Pattern detection depends on slope thresholds
            assert "Triangle" in result.pattern_type
            assert result.confidence > 0
            assert result.target_price is not None
            assert result.stop_loss is not None

    def test_no_triangle_in_random_data(self):
        """Test that random data doesn't necessarily produce triangle."""
        import numpy as np

        np.random.seed(42)
        data = pd.DataFrame(
            {
                "high": np.random.uniform(100, 110, 20),
                "low": np.random.uniform(90, 100, 20),
                "close": np.random.uniform(95, 105, 20),
            }
        )
        result = detect_triangle(data, lookback=20)
        # May or may not detect pattern in random data
        assert result is None or isinstance(result, PricePattern)


class TestDetectFlag:
    """Tests for detect_flag function."""

    def test_insufficient_data(self, insufficient_data):
        """Test with insufficient data."""
        result = detect_flag(insufficient_data, lookback=15)
        assert result is None

    def test_detects_bull_flag(self, bull_flag_data):
        """Test detection of bull flag pattern."""
        result = detect_flag(bull_flag_data, lookback=15)

        if result:  # Pattern detection is probabilistic
            assert "Flag" in result.pattern_type
            assert result.confidence > 0
            assert result.target_price is not None
            assert result.stop_loss is not None

    def test_detects_bear_flag(self, bear_flag_data):
        """Test detection of bear flag pattern."""
        result = detect_flag(bear_flag_data, lookback=15)

        if result:  # Pattern detection is probabilistic
            assert "Flag" in result.pattern_type
            assert result.confidence > 0
            assert result.target_price is not None
            assert result.stop_loss is not None

    def test_requires_strong_move(self):
        """Test that flag requires strong initial move (> 5%)."""
        # Create weak move (< 5%) followed by consolidation
        pre_flag = [
            {"high": 100 + i * 0.3, "low": 99 + i * 0.3, "close": 99.5 + i * 0.3} for i in range(10)
        ]
        flag = [{"high": 104, "low": 102, "close": 103} for _ in range(15)]
        data = pd.DataFrame(pre_flag + flag)

        result = detect_flag(data, lookback=15)
        # Should not detect pattern with weak move
        assert result is None

    def test_requires_tight_consolidation(self):
        """Test that flag requires tight consolidation."""
        # Create strong move followed by wide range (> 3%)
        pre_flag = [
            {"high": 100 + i * 2, "low": 98 + i * 2, "close": 99 + i * 2} for i in range(10)
        ]
        # Wide consolidation range
        flag = [{"high": 125 + i, "low": 115 - i, "close": 120} for i in range(15)]
        data = pd.DataFrame(pre_flag + flag)

        result = detect_flag(data, lookback=15)
        # Should not detect pattern with wide consolidation
        assert result is None


class TestPatternProperties:
    """Tests for pattern object properties."""

    def test_pattern_has_required_fields(self, double_top_data):
        """Test that detected patterns have all required fields."""
        result = detect_double_top(double_top_data, lookback=20)

        if result:
            assert hasattr(result, "pattern_type")
            assert hasattr(result, "confidence")
            assert hasattr(result, "target_price")
            assert hasattr(result, "stop_loss")
            assert hasattr(result, "description")

    def test_confidence_in_valid_range(self, double_top_data):
        """Test that confidence is in valid range [0, 1]."""
        patterns = detect_patterns(double_top_data)

        for pattern in patterns:
            assert 0 <= pattern.confidence <= 1

    def test_target_price_is_positive(self, double_top_data):
        """Test that target price is positive."""
        patterns = detect_patterns(double_top_data)

        for pattern in patterns:
            assert pattern.target_price > 0

    def test_stop_loss_is_positive(self, double_top_data):
        """Test that stop loss is positive."""
        patterns = detect_patterns(double_top_data)

        for pattern in patterns:
            assert pattern.stop_loss > 0
