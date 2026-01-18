"""Tests for `OrderService.get_order`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Order
from gpt_trader.features.brokerages.coinbase.errors import OrderQueryError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestGetOrder(OrderServiceTestBase):
    """Tests for get_order method."""

    def test_get_order_returns_order(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test that get_order returns an Order object."""
        mock_client.get_order_historical.return_value = {"order": sample_order_response}

        result = order_service.get_order("order-123")

        assert isinstance(result, Order)
        mock_client.get_order_historical.assert_called_once_with("order-123")

    def test_get_order_not_found(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that None is returned when order not found."""
        mock_client.get_order_historical.return_value = {"order": None}

        result = order_service.get_order("nonexistent-order")

        assert result is None

    def test_get_order_handles_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions raise OrderQueryError."""
        mock_client.get_order_historical.side_effect = RuntimeError("API error")

        with pytest.raises(OrderQueryError, match="Failed to get order"):
            order_service.get_order("order-123")
