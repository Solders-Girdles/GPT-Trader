"""Tests for `OrderService.cancel_order`."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.features.brokerages.coinbase.errors import OrderCancellationError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestCancelOrder(OrderServiceTestBase):
    """Tests for cancel_order method."""

    def test_cancel_order_success(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test successful order cancellation."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "order-123", "success": True}]
        }

        result = order_service.cancel_order("order-123")

        assert result is True
        mock_client.cancel_orders.assert_called_once_with(order_ids=["order-123"])

    def test_cancel_order_failure(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test failed order cancellation raises OrderCancellationError."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "order-123", "success": False}]
        }

        with pytest.raises(OrderCancellationError, match="Cancellation rejected"):
            order_service.cancel_order("order-123")

    def test_cancel_order_not_found(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test cancellation raises when order not in results."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "other-order", "success": True}]
        }

        with pytest.raises(OrderCancellationError, match="not found in cancellation response"):
            order_service.cancel_order("order-123")

    def test_cancel_order_empty_results(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test cancellation with empty results raises OrderCancellationError."""
        mock_client.cancel_orders.return_value = {"results": []}

        with pytest.raises(OrderCancellationError, match="not found in cancellation response"):
            order_service.cancel_order("order-123")

    def test_cancel_order_handles_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions raise OrderCancellationError."""
        mock_client.cancel_orders.side_effect = RuntimeError("API error")

        with pytest.raises(OrderCancellationError, match="Unexpected error"):
            order_service.cancel_order("order-123")
