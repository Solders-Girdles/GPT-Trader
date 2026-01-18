"""Tests for `OrderService.list_orders`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Order
from gpt_trader.features.brokerages.coinbase.errors import OrderQueryError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestListOrders(OrderServiceTestBase):
    """Tests for list_orders method."""

    def test_list_orders_returns_orders(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test that list_orders returns Order objects."""
        mock_client.list_orders.return_value = {
            "orders": [sample_order_response],
            "cursor": None,
        }

        result = order_service.list_orders()

        assert len(result) == 1
        assert isinstance(result[0], Order)

    def test_list_orders_with_product_filter(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by product_id."""
        mock_client.list_orders.return_value = {"orders": [], "cursor": None}

        order_service.list_orders(product_id="BTC-USD")

        call_kwargs = mock_client.list_orders.call_args.kwargs
        assert call_kwargs["product_id"] == "BTC-USD"

    def test_list_orders_with_status_filter(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by status."""
        mock_client.list_orders.return_value = {"orders": [], "cursor": None}

        order_service.list_orders(status=["PENDING", "OPEN"])

        call_kwargs = mock_client.list_orders.call_args.kwargs
        assert call_kwargs["order_status"] == ["PENDING", "OPEN"]

    def test_list_orders_pagination(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test that pagination is handled correctly."""
        page1_response = sample_order_response.copy()
        page1_response["order_id"] = "order-1"
        page2_response = sample_order_response.copy()
        page2_response["order_id"] = "order-2"

        mock_client.list_orders.side_effect = [
            {"orders": [page1_response], "cursor": "cursor-123"},
            {"orders": [page2_response], "cursor": None},
        ]

        result = order_service.list_orders()

        assert len(result) == 2
        assert mock_client.list_orders.call_count == 2

    def test_list_orders_handles_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions raise OrderQueryError."""
        mock_client.list_orders.side_effect = RuntimeError("API error")

        with pytest.raises(OrderQueryError, match="Failed to list orders"):
            order_service.list_orders()

    def test_list_orders_respects_limit(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that limit is passed to API."""
        mock_client.list_orders.return_value = {"orders": [], "cursor": None}

        order_service.list_orders(limit=50)

        call_kwargs = mock_client.list_orders.call_args.kwargs
        assert call_kwargs["limit"] == 50
