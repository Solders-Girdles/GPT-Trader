"""Tests for data quality checking functionality."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.data.quality import DataQualityChecker
from bot_v2.features.data.types import DataQuality


@pytest.fixture
def quality_checker():
    """Create a DataQualityChecker instance."""
    return DataQualityChecker()


@pytest.fixture
def valid_ohlcv_data():
    """Create valid OHLCV data."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(30)],
            "high": [101 + i * 0.5 for i in range(30)],
            "low": [99 + i * 0.5 for i in range(30)],
            "close": [100.5 + i * 0.5 for i in range(30)],
            "volume": [1000000 + i * 1000 for i in range(30)],
        },
        index=dates,
    )


@pytest.fixture
def invalid_ohlcv_data():
    """Create invalid OHLCV data with various issues."""
    dates = pd.date_range(end=datetime.now(), periods=10, freq="D")
    return pd.DataFrame(
        {
            "open": [100, 101, 102, None, 104, 105, 106, 107, 108, 109],
            "high": [99, 102, 103, 104, 105, 106, 107, 108, 109, 110],  # First row: high < low
            "low": [100, 100, 101, 102, 103, 104, 105, 106, 107, 108],  # First row: low > open
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [
                1000000,
                -5000,
                1000000,
                1000000,
                1000000,
                1000000,
                1000000,
                1000000,
                1000000,
                1000000,
            ],
        },
        index=dates,
    )


@pytest.fixture
def old_data():
    """Create old data for timeliness testing."""
    old_dates = pd.date_range(end=datetime.now() - timedelta(days=100), periods=30, freq="D")
    return pd.DataFrame(
        {
            "open": [100 + i * 0.5 for i in range(30)],
            "high": [101 + i * 0.5 for i in range(30)],
            "low": [99 + i * 0.5 for i in range(30)],
            "close": [100.5 + i * 0.5 for i in range(30)],
            "volume": [1000000 for _ in range(30)],
        },
        index=old_dates,
    )


class TestDataQualityCheckerInit:
    """Tests for DataQualityChecker initialization."""

    def test_init(self):
        """Test initialization creates empty history."""
        checker = DataQualityChecker()
        assert checker.quality_history == []
        assert checker.max_history == 100


class TestCheckQuality:
    """Tests for check_quality method."""

    def test_empty_dataframe(self, quality_checker):
        """Test quality check on empty DataFrame."""
        result = quality_checker.check_quality(pd.DataFrame())

        assert result.completeness == 0.0
        assert result.accuracy == 0.0
        assert result.consistency == 0.0
        assert result.timeliness == 0.0

    def test_valid_data(self, quality_checker, valid_ohlcv_data):
        """Test quality check on valid data."""
        result = quality_checker.check_quality(valid_ohlcv_data)

        assert isinstance(result, DataQuality)
        assert result.completeness > 0.9  # Should have high completeness
        assert result.accuracy > 0.9  # Should have high accuracy
        assert result.consistency > 0.7  # Should have good consistency

    def test_quality_stored_in_history(self, quality_checker, valid_ohlcv_data):
        """Test that quality is stored in history."""
        assert len(quality_checker.quality_history) == 0

        quality_checker.check_quality(valid_ohlcv_data)
        assert len(quality_checker.quality_history) == 1

        quality_checker.check_quality(valid_ohlcv_data)
        assert len(quality_checker.quality_history) == 2

    def test_history_max_size(self, quality_checker, valid_ohlcv_data):
        """Test that history respects max size."""
        # Fill beyond max history
        for _ in range(150):
            quality_checker.check_quality(valid_ohlcv_data)

        assert len(quality_checker.quality_history) == quality_checker.max_history

    def test_completeness_calculation(self, quality_checker):
        """Test completeness calculation with missing values."""
        data = pd.DataFrame(
            {
                "open": [100, None, 102, None, 104],
                "close": [101, 102, 103, 104, 105],
            }
        )

        result = quality_checker.check_quality(data)
        # 8 non-null out of 10 total values = 0.8
        assert result.completeness == pytest.approx(0.8, abs=0.01)


class TestCheckAccuracy:
    """Tests for _check_accuracy method."""

    def test_non_ohlc_data(self, quality_checker):
        """Test accuracy check on non-OHLC data."""
        data = pd.DataFrame({"price": [100, 101, 102]})
        result = quality_checker._check_accuracy(data)
        assert result == 1.0  # Should assume accurate for non-OHLC

    def test_valid_ohlc_relationships(self, quality_checker, valid_ohlcv_data):
        """Test accuracy check on valid OHLC relationships."""
        result = quality_checker._check_accuracy(valid_ohlcv_data)
        assert result > 0.95  # Should have very high accuracy

    def test_invalid_high_low(self, quality_checker):
        """Test detection of high < low."""
        data = pd.DataFrame(
            {
                "high": [100, 90],  # Second row: high < low
                "low": [99, 95],
            }
        )
        result = quality_checker._check_accuracy(data)
        assert result < 1.0  # Should detect issue

    def test_negative_prices(self, quality_checker):
        """Test detection of negative prices."""
        data = pd.DataFrame(
            {
                "open": [100, -50],
                "high": [101, 51],
                "low": [99, -51],
                "close": [100.5, 50],
            }
        )
        result = quality_checker._check_accuracy(data)
        assert result < 1.0  # Should detect negative prices

    def test_high_not_highest(self, quality_checker):
        """Test detection when high is not the highest."""
        data = pd.DataFrame(
            {
                "open": [100, 100],
                "high": [101, 101],
                "low": [99, 99],
                "close": [102, 100],  # First row: close > high
            }
        )
        result = quality_checker._check_accuracy(data)
        assert result < 1.0  # Should detect issue


class TestCheckConsistency:
    """Tests for _check_consistency method."""

    def test_non_close_data(self, quality_checker):
        """Test consistency check on data without close column."""
        data = pd.DataFrame({"open": [100, 101, 102]})
        result = quality_checker._check_consistency(data)
        assert result == 1.0

    def test_single_row(self, quality_checker):
        """Test consistency check on single row."""
        data = pd.DataFrame({"close": [100]})
        result = quality_checker._check_consistency(data)
        assert result == 1.0  # No returns to check

    def test_reasonable_returns(self, quality_checker):
        """Test consistency check with reasonable returns."""
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104, 105]})
        result = quality_checker._check_consistency(data)
        assert result > 0.8  # Should have good consistency

    def test_extreme_returns(self, quality_checker):
        """Test detection of extreme returns."""
        data = pd.DataFrame({"close": [100, 200, 100, 200]})  # 100% daily returns
        result = quality_checker._check_consistency(data)
        assert result < 0.8  # Should detect suspicious movement

    def test_no_variance(self, quality_checker):
        """Test detection of zero variance."""
        data = pd.DataFrame({"close": [100] * 10})
        result = quality_checker._check_consistency(data)
        assert result < 1.0  # Should penalize for no movement

    def test_datetime_index_coverage(self, quality_checker):
        """Test coverage check with datetime index."""
        # Create data with gaps
        dates = pd.bdate_range(start="2023-01-01", periods=10, freq="2D")  # Every 2 business days
        data = pd.DataFrame({"close": [100 + i for i in range(10)]}, index=dates)

        result = quality_checker._check_consistency(data)
        # Result will be lower due to gaps in expected business days
        assert 0 <= result <= 1.0


class TestCheckTimeliness:
    """Tests for _check_timeliness method."""

    def test_empty_data(self, quality_checker):
        """Test timeliness check on empty DataFrame."""
        result = quality_checker._check_timeliness(pd.DataFrame())
        assert result == 0.0

    def test_recent_data(self, quality_checker):
        """Test timeliness check on recent data."""
        data = pd.DataFrame(
            {"close": [100, 101, 102]}, index=pd.date_range(end=datetime.now(), periods=3, freq="D")
        )
        result = quality_checker._check_timeliness(data)
        assert result >= 0.95  # Should have high timeliness for today's data

    def test_old_data(self, quality_checker, old_data):
        """Test timeliness check on old data."""
        result = quality_checker._check_timeliness(old_data)
        # 100 days old: 1.0 - (100/365) ≈ 0.726
        assert 0.7 < result < 0.75  # Should have moderate timeliness for 100-day old data

    def test_very_old_data(self, quality_checker):
        """Test timeliness check on very old data."""
        old_dates = pd.date_range(end=datetime.now() - timedelta(days=400), periods=10, freq="D")
        data = pd.DataFrame({"close": [100] * 10}, index=old_dates)

        result = quality_checker._check_timeliness(data)
        assert result >= 0.1  # Should have minimum score
        assert result < 0.5

    def test_non_datetime_index(self, quality_checker):
        """Test timeliness check on non-datetime index."""
        data = pd.DataFrame({"close": [100, 101, 102]})
        result = quality_checker._check_timeliness(data)
        assert result == 0.5  # Default score


class TestValidateOHLCV:
    """Tests for validate_ohlcv method."""

    def test_valid_data(self, quality_checker, valid_ohlcv_data):
        """Test validation of valid OHLCV data."""
        issues = quality_checker.validate_ohlcv(valid_ohlcv_data)
        assert issues == []

    def test_missing_columns(self, quality_checker):
        """Test detection of missing columns."""
        data = pd.DataFrame({"open": [100], "close": [101]})
        issues = quality_checker.validate_ohlcv(data)

        assert len(issues) > 0
        assert any("Missing columns" in issue for issue in issues)

    def test_null_values(self, quality_checker, invalid_ohlcv_data):
        """Test detection of null values."""
        issues = quality_checker.validate_ohlcv(invalid_ohlcv_data)

        assert any("Null values" in issue for issue in issues)

    def test_invalid_high_low(self, quality_checker, invalid_ohlcv_data):
        """Test detection of high < low."""
        issues = quality_checker.validate_ohlcv(invalid_ohlcv_data)

        assert any("High < Low" in issue for issue in issues)

    def test_invalid_high_relationship(self, quality_checker):
        """Test detection of high below open/close."""
        data = pd.DataFrame(
            {
                "open": [100],
                "high": [99],  # high < open
                "low": [98],
                "close": [99.5],
                "volume": [1000],
            }
        )
        issues = quality_checker.validate_ohlcv(data)

        assert any("High below" in issue for issue in issues)

    def test_invalid_low_relationship(self, quality_checker):
        """Test detection of low above open/close."""
        data = pd.DataFrame(
            {
                "open": [100],
                "high": [102],
                "low": [101],  # low > open
                "close": [100.5],
                "volume": [1000],
            }
        )
        issues = quality_checker.validate_ohlcv(data)

        assert any("Low above" in issue for issue in issues)

    def test_negative_prices(self, quality_checker):
        """Test detection of negative prices."""
        data = pd.DataFrame(
            {
                "open": [-100],
                "high": [101],
                "low": [99],
                "close": [100],
                "volume": [1000],
            }
        )
        issues = quality_checker.validate_ohlcv(data)

        assert any("Negative" in issue and "open" in issue for issue in issues)

    def test_negative_volume(self, quality_checker, invalid_ohlcv_data):
        """Test detection of negative volume."""
        issues = quality_checker.validate_ohlcv(invalid_ohlcv_data)

        assert any("Negative volume" in issue for issue in issues)


class TestGetQualityTrend:
    """Tests for get_quality_trend method."""

    def test_empty_history(self, quality_checker):
        """Test quality trend with empty history."""
        trend = quality_checker.get_quality_trend()

        assert trend["completeness"] == 0.0
        assert trend["accuracy"] == 0.0
        assert trend["consistency"] == 0.0
        assert trend["timeliness"] == 0.0

    def test_single_quality_check(self, quality_checker, valid_ohlcv_data):
        """Test quality trend with single check."""
        quality = quality_checker.check_quality(valid_ohlcv_data)
        trend = quality_checker.get_quality_trend()

        assert trend["completeness"] == pytest.approx(quality.completeness, abs=0.01)
        assert trend["accuracy"] == pytest.approx(quality.accuracy, abs=0.01)
        assert trend["consistency"] == pytest.approx(quality.consistency, abs=0.01)
        assert trend["timeliness"] == pytest.approx(quality.timeliness, abs=0.01)

    def test_multiple_quality_checks(self, quality_checker, valid_ohlcv_data):
        """Test quality trend with multiple checks."""
        # Run multiple checks
        for _ in range(5):
            quality_checker.check_quality(valid_ohlcv_data)

        trend = quality_checker.get_quality_trend()

        # All metrics should be positive
        assert trend["completeness"] > 0
        assert trend["accuracy"] > 0
        assert trend["consistency"] > 0
        assert trend["timeliness"] > 0

    def test_trend_averages_over_history(self, quality_checker):
        """Test that trend averages over all history."""
        # Create good data
        good_data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
            }
        )

        # Create poor data
        poor_data = pd.DataFrame(
            {
                "close": [100, None, None],
                "high": [101, None, None],
                "low": [99, None, None],
            }
        )

        # Check both
        quality_checker.check_quality(good_data)
        quality_checker.check_quality(poor_data)

        trend = quality_checker.get_quality_trend()

        # Trend should be average of both checks
        assert 0 < trend["completeness"] < 1.0
