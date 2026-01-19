"""Tests for `OrderService.list_fills`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.features.brokerages.coinbase.errors import BrokerageError, OrderQueryError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestListFills(OrderServiceTestBase):
    """Tests for list_fills method."""

    def test_list_fills_returns_fills(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that list_fills returns fill data."""
        fill_data = {
            "fill_id": "fill-123",
            "order_id": "order-123",
            "product_id": "BTC-USD",
            "price": "50000",
            "size": "0.1",
        }
        mock_client.list_fills.return_value = {"fills": [fill_data], "cursor": None}

        result = order_service.list_fills()

        assert len(result) == 1
        assert result[0]["fill_id"] == "fill-123"

    def test_list_fills_with_product_filter(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by product_id."""
        mock_client.list_fills.return_value = {"fills": [], "cursor": None}

        order_service.list_fills(product_id="ETH-USD")

        call_kwargs = mock_client.list_fills.call_args.kwargs
        assert call_kwargs["product_id"] == "ETH-USD"

    def test_list_fills_with_order_filter(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by order_id."""
        mock_client.list_fills.return_value = {"fills": [], "cursor": None}

        order_service.list_fills(order_id="order-456")

        call_kwargs = mock_client.list_fills.call_args.kwargs
        assert call_kwargs["order_id"] == "order-456"

    def test_list_fills_pagination(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that pagination is handled correctly."""
        mock_client.list_fills.side_effect = [
            {"fills": [{"fill_id": "fill-1"}], "cursor": "cursor-123"},
            {"fills": [{"fill_id": "fill-2"}], "cursor": None},
        ]

        result = order_service.list_fills()

        assert len(result) == 2
        assert mock_client.list_fills.call_count == 2

    def test_list_fills_handles_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions raise OrderQueryError."""
        mock_client.list_fills.side_effect = RuntimeError("API error")

        with pytest.raises(OrderQueryError, match="Failed to list fills"):
            order_service.list_fills()

    def test_list_fills_re_raises_brokerage_error(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that BrokerageError isn't wrapped."""
        mock_client.list_fills.side_effect = BrokerageError("rate limited")

        with pytest.raises(BrokerageError, match="rate limited"):
            order_service.list_fills()
