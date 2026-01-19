"""Tests for `OrderService` edge cases."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.brokerages.coinbase.errors import OrderCancellationError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestOrderServiceEdgeCases(OrderServiceTestBase):
    """Tests for edge cases."""

    def test_list_orders_empty_response(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test handling of empty orders response."""
        mock_client.list_orders.return_value = {"orders": []}

        result = order_service.list_orders()

        assert result == []

    def test_list_fills_empty_response(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test handling of empty fills response."""
        mock_client.list_fills.return_value = {"fills": []}

        result = order_service.list_fills()

        assert result == []

    def test_cancel_order_missing_results_key(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test handling of response without results key raises error."""
        mock_client.cancel_orders.return_value = {}

        with pytest.raises(OrderCancellationError, match="not found in cancellation response"):
            order_service.cancel_order("order-123")

    def test_place_order_market_without_price(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
        mock_payload_executor: MagicMock,
    ) -> None:
        """Test placing market order without price delegates correctly."""
        mock_payload_executor.execute_order_payload.return_value = Order(
            id="order-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
        )

        call_kwargs = mock_payload_builder.build_order_payload.call_args.kwargs
        assert "price" in call_kwargs
        assert call_kwargs["price"] is None
