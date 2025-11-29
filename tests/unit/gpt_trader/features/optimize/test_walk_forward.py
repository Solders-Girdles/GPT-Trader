"""Unit tests for Walk-Forward Analysis module."""

from __future__ import annotations

from datetime import datetime

import pytest

from gpt_trader.features.optimize.walk_forward import (
    WalkForwardConfig,
    WalkForwardWindow,
    WindowResult,
    _add_months,
    _months_between,
    generate_windows,
)


class TestWalkForwardConfig:
    """Tests for WalkForwardConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = WalkForwardConfig()
        assert config.train_months == 6
        assert config.test_months == 1
        assert config.anchor_start is False
        assert config.min_trades_per_window == 10
        assert config.overlap_months == 0

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = WalkForwardConfig(
            train_months=3,
            test_months=2,
            anchor_start=True,
            min_trades_per_window=20,
        )
        assert config.train_months == 3
        assert config.test_months == 2
        assert config.anchor_start is True
        assert config.min_trades_per_window == 20

    def test_invalid_train_months_raises(self) -> None:
        """Test that invalid train_months raises ValueError."""
        with pytest.raises(ValueError, match="train_months must be at least 1"):
            WalkForwardConfig(train_months=0)

    def test_invalid_test_months_raises(self) -> None:
        """Test that invalid test_months raises ValueError."""
        with pytest.raises(ValueError, match="test_months must be at least 1"):
            WalkForwardConfig(test_months=0)

    def test_overlap_exceeds_train_raises(self) -> None:
        """Test that overlap >= train_months raises ValueError."""
        with pytest.raises(ValueError, match="overlap_months must be less than"):
            WalkForwardConfig(train_months=3, overlap_months=3)

        with pytest.raises(ValueError, match="overlap_months must be less than"):
            WalkForwardConfig(train_months=3, overlap_months=5)


class TestWalkForwardWindow:
    """Tests for WalkForwardWindow dataclass."""

    def test_window_creation(self) -> None:
        """Test basic window creation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )
        assert window.window_id == 0
        assert window.train_start == datetime(2024, 1, 1)
        assert window.train_end == datetime(2024, 4, 1)
        assert window.test_start == datetime(2024, 4, 1)
        assert window.test_end == datetime(2024, 5, 1)

    def test_train_days_property(self) -> None:
        """Test train_days property calculation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),  # ~90 days
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )
        assert window.train_days == 91  # Jan + Feb + Mar = 31 + 29 + 31 = 91 (2024 leap year)

    def test_test_days_property(self) -> None:
        """Test test_days property calculation."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),  # 30 days in April
        )
        assert window.test_days == 30


class TestAddMonths:
    """Tests for _add_months helper function."""

    def test_add_months_simple(self) -> None:
        """Test adding months within same year."""
        result = _add_months(datetime(2024, 1, 15), 3)
        assert result == datetime(2024, 4, 15)

    def test_add_months_cross_year(self) -> None:
        """Test adding months across year boundary."""
        result = _add_months(datetime(2024, 11, 15), 3)
        assert result == datetime(2025, 2, 15)

    def test_add_months_day_overflow(self) -> None:
        """Test handling of day overflow (e.g., Jan 31 + 1 month)."""
        # January 31 + 1 month should become February 29 (2024 is leap year)
        result = _add_months(datetime(2024, 1, 31), 1)
        assert result == datetime(2024, 2, 29)

        # March 31 + 1 month should become April 30
        result = _add_months(datetime(2024, 3, 31), 1)
        assert result == datetime(2024, 4, 30)

    def test_add_twelve_months(self) -> None:
        """Test adding 12 months equals one year."""
        result = _add_months(datetime(2024, 6, 15), 12)
        assert result == datetime(2025, 6, 15)


class TestMonthsBetween:
    """Tests for _months_between helper function."""

    def test_one_month(self) -> None:
        """Test approximately one month difference."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 2, 1)
        months = _months_between(start, end)
        assert 0.9 < months < 1.1

    def test_six_months(self) -> None:
        """Test approximately six months difference."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 7, 1)
        months = _months_between(start, end)
        assert 5.9 < months < 6.1

    def test_twelve_months(self) -> None:
        """Test approximately twelve months difference."""
        start = datetime(2024, 1, 1)
        end = datetime(2025, 1, 1)
        months = _months_between(start, end)
        assert 11.9 < months < 12.1


class TestGenerateWindows:
    """Tests for generate_windows function."""

    def test_basic_rolling_window(self) -> None:
        """Test basic rolling window generation."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        # Should generate multiple windows
        assert len(windows) >= 8  # 12 - 3 - 1 = 8 possible windows

        # First window should start at the beginning
        assert windows[0].train_start == datetime(2024, 1, 1)
        assert windows[0].train_end == datetime(2024, 4, 1)
        assert windows[0].test_start == datetime(2024, 4, 1)
        assert windows[0].test_end == datetime(2024, 5, 1)

    def test_anchored_window(self) -> None:
        """Test anchored (expanding) window generation."""
        config = WalkForwardConfig(train_months=3, test_months=1, anchor_start=True)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 8, 1),
            config=config,
        )

        # All windows should start from the beginning (anchored)
        for window in windows:
            assert window.train_start == datetime(2024, 1, 1)

    def test_window_continuity(self) -> None:
        """Test that windows are continuous (no gaps between test end and next train start)."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        # Verify test windows are continuous for rolling windows
        for i in range(len(windows) - 1):
            current_test_end = windows[i].test_end
            next_test_start = windows[i + 1].test_start
            # Next test should start one month after current test
            assert _months_between(current_test_end, next_test_start) < 0.1

    def test_insufficient_data_raises(self) -> None:
        """Test that insufficient data raises ValueError."""
        config = WalkForwardConfig(train_months=6, test_months=1)

        # Only 5 months of data, need at least 7 (6 train + 1 test)
        with pytest.raises(ValueError, match="Date range too short"):
            generate_windows(
                start_date=datetime(2024, 1, 1),
                end_date=datetime(2024, 5, 1),
                config=config,
            )

    def test_exactly_one_window(self) -> None:
        """Test generating exactly one window."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 5, 1),  # Exactly 4 months: 3 train + 1 test
            config=config,
        )

        assert len(windows) == 1
        assert windows[0].train_start == datetime(2024, 1, 1)
        assert windows[0].test_end == datetime(2024, 5, 1)

    def test_window_ids_sequential(self) -> None:
        """Test that window IDs are sequential starting from 0."""
        config = WalkForwardConfig(train_months=2, test_months=1)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        for i, window in enumerate(windows):
            assert window.window_id == i

    def test_test_follows_train(self) -> None:
        """Test that test period immediately follows train period."""
        config = WalkForwardConfig(train_months=4, test_months=2)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        for window in windows:
            assert window.test_start == window.train_end


class TestWindowResult:
    """Tests for WindowResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid window result."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )

        result = WindowResult(
            window=window,
            best_parameters={"rsi_period": 14, "ma_period": 20},
            optimization_trials=50,
            best_train_objective=1.5,
            test_objective_value=1.2,
            is_valid=True,
        )

        assert result.is_valid is True
        assert result.best_parameters["rsi_period"] == 14
        assert result.test_objective_value == 1.2

    def test_invalid_result_with_errors(self) -> None:
        """Test creating an invalid window result with errors."""
        window = WalkForwardWindow(
            window_id=0,
            train_start=datetime(2024, 1, 1),
            train_end=datetime(2024, 4, 1),
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )

        result = WindowResult(
            window=window,
            best_parameters={},
            optimization_trials=50,
            best_train_objective=float("-inf"),
            is_valid=False,
            validation_errors=["No valid optimization trials completed"],
        )

        assert result.is_valid is False
        assert len(result.validation_errors) == 1


class TestEdgeCases:
    """Test edge cases for walk-forward analysis."""

    def test_large_train_window(self) -> None:
        """Test with large train window (12 months)."""
        config = WalkForwardConfig(train_months=12, test_months=1)
        windows = generate_windows(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        # Should generate windows
        assert len(windows) >= 1
        assert windows[0].train_days >= 365

    def test_large_test_window(self) -> None:
        """Test with large test window (3 months)."""
        config = WalkForwardConfig(train_months=6, test_months=3)
        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            config=config,
        )

        # Should generate fewer windows due to larger test period
        assert len(windows) >= 1
        for window in windows:
            assert window.test_days >= 28 * 3  # At least ~3 months

    def test_windows_dont_exceed_end_date(self) -> None:
        """Test that no window test period exceeds the end date."""
        config = WalkForwardConfig(train_months=3, test_months=1)
        end_date = datetime(2024, 12, 15)  # Mid-month end

        windows = generate_windows(
            start_date=datetime(2024, 1, 1),
            end_date=end_date,
            config=config,
        )

        for window in windows:
            assert window.test_end <= end_date
