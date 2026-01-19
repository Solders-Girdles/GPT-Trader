"""Tests for `OrderService.place_order`."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestPlaceOrder(OrderServiceTestBase):
    """Tests for place_order method."""

    def test_place_order_builds_payload(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
    ) -> None:
        """Test that place_order delegates to payload builder."""
        order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
        )

        mock_payload_builder.build_order_payload.assert_called_once()
        call_kwargs = mock_payload_builder.build_order_payload.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-USD"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["quantity"] == Decimal("1.5")
        assert call_kwargs["price"] == Decimal("50000")

    def test_place_order_executes_payload(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
        mock_payload_executor: MagicMock,
    ) -> None:
        """Test that place_order executes the built payload."""
        mock_payload = {"product_id": "ETH-USD"}
        mock_payload_builder.build_order_payload.return_value = mock_payload

        mock_order = Order(
            id="order-123",
            symbol="ETH-USD",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_payload_executor.execute_order_payload.return_value = mock_order

        result = order_service.place_order(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
        )

        mock_payload_executor.execute_order_payload.assert_called_once_with(
            "ETH-USD", mock_payload, None
        )
        assert result == mock_order

    def test_place_order_passes_all_parameters(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
    ) -> None:
        """Test that all optional parameters are passed correctly."""
        order_service.place_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=Decimal("49000"),
            tif=TimeInForce.IOC,
            client_id="my-order-123",
            reduce_only=True,
            leverage=10,
            post_only=True,
        )

        mock_payload_builder.build_order_payload.assert_called_once()
        call_kwargs = mock_payload_builder.build_order_payload.call_args.kwargs
        assert call_kwargs["client_id"] == "my-order-123"
        assert call_kwargs["stop_price"] == Decimal("49000")
        assert call_kwargs["tif"] == TimeInForce.IOC
        assert call_kwargs["reduce_only"] is True
        assert call_kwargs["leverage"] == 10
        assert call_kwargs["post_only"] is True

    def test_place_order_returns_order(
        self,
        order_service: OrderService,
        mock_payload_executor: MagicMock,
    ) -> None:
        """Test that place_order returns an Order object."""
        mock_order = Order(
            id="order-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        mock_payload_executor.execute_order_payload.return_value = mock_order

        result = order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        assert isinstance(result, Order)
        assert result.id == "order-123"
