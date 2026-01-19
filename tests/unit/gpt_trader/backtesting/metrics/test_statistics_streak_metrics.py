"""Tests for trade statistics streak calculations."""

from __future__ import annotations

from tests.unit.gpt_trader.backtesting.metrics.statistics_test_helpers import _create_mock_order

from gpt_trader.backtesting.metrics.statistics import _calculate_streak_metrics


class TestCalculateStreakMetrics:
    """Tests for _calculate_streak_metrics function."""

    def test_returns_zeros_placeholder(self) -> None:
        result = _calculate_streak_metrics([])
        assert result["max_wins"] == 0
        assert result["max_losses"] == 0
        assert result["current"] == 0

    def test_with_orders_returns_zeros(self) -> None:
        orders = [_create_mock_order() for _ in range(5)]
        result = _calculate_streak_metrics(orders)
        assert result["max_wins"] == 0
        assert result["max_losses"] == 0
        assert result["current"] == 0
