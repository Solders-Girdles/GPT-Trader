"""Unit tests for walk-forward time helper functions."""

from __future__ import annotations

from datetime import datetime

from gpt_trader.features.optimize.walk_forward import _add_months, _months_between


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
