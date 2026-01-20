"""Contract tests for Coinbase REST OrderService query and fill APIs."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.order_service as order_service_module
from gpt_trader.core import Order
from gpt_trader.features.brokerages.coinbase.errors import OrderQueryError
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.contract_suite_test_base import (
    CoinbaseRestContractSuiteBase,
)


class TestCoinbaseRestContractSuiteOrderServiceQueriesAndFills(CoinbaseRestContractSuiteBase):
    def test_list_orders_with_pagination(
        self, order_service, mock_client, monkeypatch: pytest.MonkeyPatch
    ):
        """Test order listing with pagination handling."""
        mock_client.list_orders.side_effect = [
            {"orders": [{"order_id": "1"}, {"order_id": "2"}], "cursor": "next_page"},
            {"orders": [{"order_id": "3"}]},
        ]

        monkeypatch.setattr(order_service_module, "to_order", lambda x: Mock(spec=Order))
        orders = order_service.list_orders()

        assert len(orders) == 3
        assert mock_client.list_orders.call_count == 2

    def test_list_orders_error_handling(self, order_service, mock_client):
        """Test order listing raises OrderQueryError on error."""
        mock_client.list_orders.side_effect = Exception("API error")

        with pytest.raises(OrderQueryError, match="Failed to list orders"):
            order_service.list_orders()

    def test_get_order_success(self, order_service, mock_client, monkeypatch: pytest.MonkeyPatch):
        """Test successful order retrieval."""
        mock_order_data = {"order_id": "test_123", "status": "filled"}
        mock_client.get_order_historical.return_value = {"order": mock_order_data}

        monkeypatch.setattr(order_service_module, "to_order", lambda x: Mock(spec=Order))
        order = order_service.get_order("test_123")

        assert order is not None

    def test_get_order_not_found(self, order_service, mock_client):
        """Test order retrieval raises OrderQueryError when exception occurs."""
        mock_client.get_order_historical.side_effect = Exception("Order not found")

        with pytest.raises(OrderQueryError, match="Failed to get order"):
            order_service.get_order("test_123")

    def test_list_fills_with_pagination(self, order_service, mock_client):
        """Test fills listing with pagination."""
        mock_client.list_fills.side_effect = [
            {"fills": [{"fill_id": "1"}, {"fill_id": "2"}], "cursor": "next"},
            {"fills": [{"fill_id": "3"}]},
        ]

        fills = order_service.list_fills()

        assert len(fills) == 3

    def test_list_fills_error_handling(self, order_service, mock_client):
        """Test fills listing raises OrderQueryError on error."""
        mock_client.list_fills.side_effect = Exception("API error")

        with pytest.raises(OrderQueryError, match="Failed to list fills"):
            order_service.list_fills()
