"""Tests for trade statistics PnL calculations."""

from __future__ import annotations

from decimal import Decimal

from tests.unit.gpt_trader.backtesting.metrics.statistics_test_helpers import _create_mock_order

from gpt_trader.backtesting.metrics.statistics import _calculate_pnl_metrics


class TestCalculatePnlMetrics:
    """Tests for _calculate_pnl_metrics function."""

    def test_empty_orders_returns_zeros(self) -> None:
        result = _calculate_pnl_metrics([])
        assert result["total_pnl"] == Decimal("0")
        assert result["gross_profit"] == Decimal("0")
        assert result["gross_loss"] == Decimal("0")
        assert result["largest_win"] == Decimal("0")
        assert result["largest_loss"] == Decimal("0")

    def test_orders_without_fill_price_skipped(self) -> None:
        order = _create_mock_order(avg_fill_price=None)
        result = _calculate_pnl_metrics([order])
        assert result["total_pnl"] == Decimal("0")

    def test_returns_structure_correct(self) -> None:
        orders = [_create_mock_order()]
        result = _calculate_pnl_metrics(orders)
        assert "total_pnl" in result
        assert "gross_profit" in result
        assert "gross_loss" in result
        assert "largest_win" in result
        assert "largest_loss" in result
