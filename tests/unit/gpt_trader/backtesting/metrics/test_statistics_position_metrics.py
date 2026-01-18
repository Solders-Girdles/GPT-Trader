"""Tests for trade statistics position calculations."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.backtesting.metrics.statistics_test_helpers import _create_mock_order

from gpt_trader.backtesting.metrics.statistics import _calculate_position_metrics


class TestCalculatePositionMetrics:
    """Tests for _calculate_position_metrics function."""

    def test_empty_orders_returns_defaults(self) -> None:
        broker = MagicMock()
        result = _calculate_position_metrics([], broker)
        assert result["avg_size"] == Decimal("0")
        assert result["max_size"] == Decimal("0")
        assert result["avg_leverage"] == Decimal("1")
        assert result["max_leverage"] == Decimal("1")

    def test_calculates_average_position_size(self) -> None:
        orders = [
            _create_mock_order(
                filled_quantity=Decimal("1"),
                avg_fill_price=Decimal("50000"),
            ),
            _create_mock_order(
                filled_quantity=Decimal("2"),
                avg_fill_price=Decimal("50000"),
            ),
        ]
        broker = MagicMock()
        broker.positions = {}
        result = _calculate_position_metrics(orders, broker)
        assert float(result["avg_size"]) == pytest.approx(75000.0, rel=0.01)
        assert float(result["max_size"]) == pytest.approx(100000.0, rel=0.01)

    def test_extracts_leverage_from_position(self) -> None:
        order = _create_mock_order(symbol="BTC-PERP")
        broker = MagicMock()
        position = MagicMock()
        position.leverage = 5
        broker.positions = {"BTC-PERP": position}
        result = _calculate_position_metrics([order], broker)
        assert float(result["max_leverage"]) == 5.0
        assert float(result["avg_leverage"]) == 5.0

    def test_multiple_orders_different_leverage(self) -> None:
        orders = [
            _create_mock_order(symbol="BTC-PERP"),
            _create_mock_order(symbol="ETH-PERP"),
        ]
        broker = MagicMock()
        btc_pos = MagicMock()
        btc_pos.leverage = 3
        eth_pos = MagicMock()
        eth_pos.leverage = 5
        broker.positions = {"BTC-PERP": btc_pos, "ETH-PERP": eth_pos}
        result = _calculate_position_metrics(orders, broker)
        assert float(result["max_leverage"]) == 5.0
        assert float(result["avg_leverage"]) == 4.0  # (3+5)/2

    def test_orders_without_fill_price_skipped(self) -> None:
        orders = [
            _create_mock_order(avg_fill_price=None),
            _create_mock_order(
                filled_quantity=Decimal("1"),
                avg_fill_price=Decimal("10000"),
            ),
        ]
        broker = MagicMock()
        broker.positions = {}
        result = _calculate_position_metrics(orders, broker)
        assert float(result["avg_size"]) == pytest.approx(10000.0, rel=0.01)
