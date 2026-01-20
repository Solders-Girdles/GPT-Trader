"""Tests for CoinbaseRestServiceCore lookup helpers."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.base as rest_base_module
from gpt_trader.core import Order
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCoreFindExistingOrderByClientId(RestServiceCoreTestBase):
    def test_find_existing_order_by_client_id_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_order_data = {
            "order_id": "order_123",
            "client_order_id": "client_123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        self.client.list_orders.return_value = {"orders": [mock_order_data]}
        mock_order = Mock(spec=Order)
        mock_order.id = "order_123"
        mock_order.client_id = "client_123"
        mock_order.created_at = datetime(2024, 1, 1)

        monkeypatch.setattr(rest_base_module, "to_order", lambda x: mock_order)

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result == mock_order
        self.client.list_orders.assert_called_once_with(product_id="BTC-USD")

    def test_find_existing_order_by_client_id_no_client_id(self) -> None:
        result = self.service._find_existing_order_by_client_id("BTC-USD", "")

        assert result is None
        self.client.list_orders.assert_not_called()

    def test_find_existing_order_by_client_id_no_matches(self) -> None:
        self.client.list_orders.return_value = {"orders": []}

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None

    def test_find_existing_order_by_client_id_multiple_matches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order1 = {
            "order_id": "order_1",
            "client_order_id": "client_123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        order2 = {
            "order_id": "order_2",
            "client_order_id": "client_123",
            "created_at": "2024-01-01T01:00:00Z",
        }
        self.client.list_orders.return_value = {"orders": [order1, order2]}
        mock_order = Mock(spec=Order)
        mock_order.id = "order_2"
        mock_order.client_id = "client_123"
        mock_order.created_at = datetime(2024, 1, 1, 1)

        monkeypatch.setattr(rest_base_module, "to_order", lambda x: mock_order)

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result == mock_order

    def test_find_existing_order_by_client_id_api_error(self) -> None:
        self.client.list_orders.side_effect = Exception("API error")

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None

    def test_find_existing_order_by_client_id_network_error(self) -> None:
        self.client.list_orders.side_effect = ConnectionError("network down")

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None

    def test_find_existing_order_by_client_id_value_error(self) -> None:
        self.client.list_orders.side_effect = ValueError("bad payload")

        result = self.service._find_existing_order_by_client_id("BTC-USD", "client_123")

        assert result is None
