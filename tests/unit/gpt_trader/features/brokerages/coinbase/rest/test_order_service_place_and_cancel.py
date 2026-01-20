"""Tests for `OrderService.place_order` and `OrderService.cancel_order`."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import OrderSide, OrderType, TimeInForce
from gpt_trader.features.brokerages.coinbase.errors import BrokerageError, OrderCancellationError
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.order_service_test_base import (
    OrderServiceTestBase,
)


class TestPlaceOrder(OrderServiceTestBase):
    def test_place_order_builds_payload(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
    ) -> None:
        order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
        )

        call_kwargs = mock_payload_builder.build_order_payload.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-USD"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["order_type"] == OrderType.LIMIT
        assert call_kwargs["quantity"] == Decimal("1.5")
        assert call_kwargs["price"] == Decimal("50000")

    def test_place_order_executes_payload(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
        mock_payload_executor: MagicMock,
    ) -> None:
        mock_payload = {"product_id": "ETH-USD"}
        mock_payload_builder.build_order_payload.return_value = mock_payload
        expected_result = object()
        mock_payload_executor.execute_order_payload.return_value = expected_result

        result = order_service.place_order(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
        )

        mock_payload_executor.execute_order_payload.assert_called_once_with(
            "ETH-USD", mock_payload, None
        )
        assert result is expected_result

    def test_place_order_passes_all_parameters(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
    ) -> None:
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

        call_kwargs = mock_payload_builder.build_order_payload.call_args.kwargs
        assert call_kwargs["client_id"] == "my-order-123"
        assert call_kwargs["stop_price"] == Decimal("49000")
        assert call_kwargs["tif"] == TimeInForce.IOC
        assert call_kwargs["reduce_only"] is True
        assert call_kwargs["leverage"] == 10
        assert call_kwargs["post_only"] is True

    def test_place_order_market_order_includes_price_none(
        self,
        order_service: OrderService,
        mock_payload_builder: MagicMock,
    ) -> None:
        order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        call_kwargs = mock_payload_builder.build_order_payload.call_args.kwargs
        assert "price" in call_kwargs
        assert call_kwargs["price"] is None


class TestCancelOrder(OrderServiceTestBase):
    def test_cancel_order_success(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "order-123", "success": True}]
        }

        result = order_service.cancel_order("order-123")

        assert result is True
        mock_client.cancel_orders.assert_called_once_with(order_ids=["order-123"])

    @pytest.mark.parametrize(
        ("response", "match"),
        [
            (
                {"results": [{"order_id": "order-123", "success": False}]},
                "Cancellation rejected",
            ),
            (
                {"results": [{"order_id": "other-order", "success": True}]},
                "not found in cancellation response",
            ),
            ({"results": []}, "not found in cancellation response"),
            ({}, "not found in cancellation response"),
        ],
    )
    def test_cancel_order_raises_for_rejection_or_missing_order(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        response: dict,
        match: str,
    ) -> None:
        mock_client.cancel_orders.return_value = response

        with pytest.raises(OrderCancellationError, match=match):
            order_service.cancel_order("order-123")

    def test_cancel_order_re_raises_brokerage_error(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.cancel_orders.side_effect = BrokerageError("rate limited")

        with pytest.raises(BrokerageError, match="rate limited"):
            order_service.cancel_order("order-123")

    def test_cancel_order_wraps_unexpected_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
    ) -> None:
        mock_client.cancel_orders.side_effect = RuntimeError("API error")

        with pytest.raises(OrderCancellationError, match="Unexpected error"):
            order_service.cancel_order("order-123")
