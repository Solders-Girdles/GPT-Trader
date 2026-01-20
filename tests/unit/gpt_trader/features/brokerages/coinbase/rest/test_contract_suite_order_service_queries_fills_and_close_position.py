"""Contract tests for Coinbase REST OrderService query/fill/close-position flows."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.order_service as order_service_module
from gpt_trader.core import Order, Position
from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.errors import OrderQueryError
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
)


class TestCoinbaseRestContractSuiteOrderServiceQueriesFillsAndClosePosition(
    CoinbaseRestContractSuiteBase
):
    def test_list_orders_with_pagination(
        self, order_service, mock_client, monkeypatch: pytest.MonkeyPatch
    ):
        mock_client.list_orders.side_effect = [
            {"orders": [{"order_id": "1"}, {"order_id": "2"}], "cursor": "next_page"},
            {"orders": [{"order_id": "3"}]},
        ]

        monkeypatch.setattr(order_service_module, "to_order", lambda x: Mock(spec=Order))
        orders = order_service.list_orders()

        assert len(orders) == 3
        assert mock_client.list_orders.call_count == 2

    def test_list_orders_error_handling(self, order_service, mock_client):
        mock_client.list_orders.side_effect = Exception("API error")

        with pytest.raises(OrderQueryError, match="Failed to list orders"):
            order_service.list_orders()

    def test_get_order_success(self, order_service, mock_client, monkeypatch: pytest.MonkeyPatch):
        mock_order_data = {"order_id": "test_123", "status": "filled"}
        mock_client.get_order_historical.return_value = {"order": mock_order_data}

        monkeypatch.setattr(order_service_module, "to_order", lambda x: Mock(spec=Order))
        order = order_service.get_order("test_123")

        assert order is not None

    def test_get_order_not_found(self, order_service, mock_client):
        mock_client.get_order_historical.side_effect = Exception("Order not found")

        with pytest.raises(OrderQueryError, match="Failed to get order"):
            order_service.get_order("test_123")

    def test_list_fills_with_pagination(self, order_service, mock_client):
        mock_client.list_fills.side_effect = [
            {"fills": [{"fill_id": "1"}, {"fill_id": "2"}], "cursor": "next"},
            {"fills": [{"fill_id": "3"}]},
        ]

        fills = order_service.list_fills()

        assert len(fills) == 3

    def test_list_fills_error_handling(self, order_service, mock_client):
        mock_client.list_fills.side_effect = Exception("API error")

        with pytest.raises(OrderQueryError, match="Failed to list fills"):
            order_service.list_fills()

    def test_close_position_success(
        self,
        portfolio_service,
        order_service,
        mock_client,
        monkeypatch: pytest.MonkeyPatch,
    ):
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")
        monkeypatch.setattr(portfolio_service, "list_positions", lambda: [mock_position])
        mock_client.close_position.return_value = {"order": {"order_id": "close_123"}}
        monkeypatch.setattr(order_service_module, "to_order", Mock(return_value=Mock(spec=Order)))

        order = order_service.close_position("BTC-USD")

        assert order is not None

    def test_close_position_no_position(
        self, portfolio_service, order_service, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(portfolio_service, "list_positions", lambda: [])

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-USD")

    def test_close_position_fallback(
        self,
        portfolio_service,
        order_service,
        mock_product_catalog,
        mock_product,
        mock_client,
        monkeypatch: pytest.MonkeyPatch,
    ):
        mock_product_catalog.get.return_value = mock_product

        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.0")
        monkeypatch.setattr(portfolio_service, "list_positions", lambda: [mock_position])
        mock_client.close_position.side_effect = Exception("API failed")
        fallback_order = Mock(spec=Order)
        fallback_func = Mock(return_value=fallback_order)

        order = order_service.close_position("BTC-USD", fallback=fallback_func)

        assert order == fallback_order
        fallback_func.assert_called_once()
