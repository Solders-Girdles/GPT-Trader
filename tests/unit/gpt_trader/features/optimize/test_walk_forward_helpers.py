"""Unit tests for walk-forward helpers and window models."""

from __future__ import annotations

from datetime import datetime

from gpt_trader.features.optimize.walk_forward import (
    WalkForwardWindow,
    WindowResult,
    _add_months,
    _months_between,
)


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
        result = _add_months(datetime(2024, 1, 31), 1)
        assert result == datetime(2024, 2, 29)  # 2024 is leap year

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
            train_end=datetime(2024, 4, 1),  # Jan + Feb + Mar (leap-year Feb)
            test_start=datetime(2024, 4, 1),
            test_end=datetime(2024, 5, 1),
        )
        assert window.train_days == 91

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
