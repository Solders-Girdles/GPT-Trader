"""Tests for Coinbase REST order service functionality."""

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.errors import (
    OrderCancellationError,
    OrderQueryError,
)
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from gpt_trader.features.brokerages.coinbase.rest.protocols import (
    OrderPayloadBuilder,
    OrderPayloadExecutor,
    PositionProvider,
)
from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Position,
    TimeInForce,
)


class TestOrderService:
    """Test OrderService class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.client = Mock(spec=CoinbaseClient)
        self.payload_builder = Mock(spec=OrderPayloadBuilder)
        self.payload_executor = Mock(spec=OrderPayloadExecutor)
        self.position_provider = Mock(spec=PositionProvider)

        self.service = OrderService(
            client=self.client,
            payload_builder=self.payload_builder,
            payload_executor=self.payload_executor,
            position_provider=self.position_provider,
        )

    def test_service_init(self) -> None:
        """Test service initialization."""
        assert self.service._client == self.client
        assert self.service._payload_builder == self.payload_builder
        assert self.service._payload_executor == self.payload_executor
        assert self.service._position_provider == self.position_provider

    def test_place_order_delegates_to_builder_and_executor(self) -> None:
        """Test place_order delegates to payload builder and executor."""
        mock_payload = {"product_id": "BTC-USD", "side": "BUY"}
        mock_order = Mock(spec=Order)
        mock_order.order_id = "order_123"

        self.payload_builder.build_order_payload.return_value = mock_payload
        self.payload_executor.execute_order_payload.return_value = mock_order

        result = self.service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            tif=TimeInForce.GTC,
            client_id="test_123",
        )

        self.payload_builder.build_order_payload.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            price=Decimal("50000.00"),
            stop_price=None,
            tif=TimeInForce.GTC,
            client_id="test_123",
            reduce_only=False,
            leverage=None,
            post_only=False,
        )
        self.payload_executor.execute_order_payload.assert_called_once_with(
            "BTC-USD", mock_payload, "test_123"
        )
        assert result == mock_order

    def test_place_order_with_reduce_only(self) -> None:
        """Test place_order with reduce_only flag."""
        mock_payload = {"product_id": "BTC-USD", "reduce_only": True}
        mock_order = Mock(spec=Order)

        self.payload_builder.build_order_payload.return_value = mock_payload
        self.payload_executor.execute_order_payload.return_value = mock_order

        self.service.place_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            reduce_only=True,
        )

        call_kwargs = self.payload_builder.build_order_payload.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    def test_cancel_order_success(self) -> None:
        """Test successful order cancellation."""
        self.client.cancel_orders.return_value = {
            "results": [{"order_id": "order_123", "success": True}]
        }

        result = self.service.cancel_order("order_123")

        self.client.cancel_orders.assert_called_once_with(order_ids=["order_123"])
        assert result is True

    def test_cancel_order_failure(self) -> None:
        """Test failed order cancellation raises OrderCancellationError."""
        self.client.cancel_orders.return_value = {
            "results": [{"order_id": "order_123", "success": False}]
        }

        with pytest.raises(OrderCancellationError, match="Cancellation rejected"):
            self.service.cancel_order("order_123")

    def test_cancel_order_not_found_in_results(self) -> None:
        """Test cancellation raises when order not in results."""
        self.client.cancel_orders.return_value = {
            "results": [{"order_id": "other_order", "success": True}]
        }

        with pytest.raises(OrderCancellationError, match="not found in cancellation response"):
            self.service.cancel_order("order_123")

    def test_cancel_order_exception(self) -> None:
        """Test cancellation raises OrderCancellationError on exception."""
        self.client.cancel_orders.side_effect = Exception("API error")

        with pytest.raises(OrderCancellationError, match="Unexpected error"):
            self.service.cancel_order("order_123")

    def test_list_orders_returns_orders(self) -> None:
        """Test listing orders returns parsed orders."""
        self.client.list_orders.return_value = {
            "orders": [
                {
                    "order_id": "order_1",
                    "product_id": "BTC-USD",
                    "side": "BUY",
                    "status": "FILLED",
                    "order_type": "LIMIT",
                    "base_size": "0.1",
                    "filled_size": "0.1",
                    "filled_value": "5000.00",
                    "average_filled_price": "50000.00",
                    "created_time": "2024-01-01T00:00:00Z",
                }
            ],
            "cursor": None,
        }

        result = self.service.list_orders(product_id="BTC-USD")

        assert len(result) == 1
        assert result[0].id == "order_1"
        self.client.list_orders.assert_called_once()

    def test_list_orders_with_pagination(self) -> None:
        """Test listing orders handles pagination."""
        self.client.list_orders.side_effect = [
            {
                "orders": [
                    {
                        "order_id": "order_1",
                        "product_id": "BTC-USD",
                        "side": "BUY",
                        "status": "FILLED",
                        "order_type": "LIMIT",
                        "base_size": "0.1",
                        "filled_size": "0.1",
                        "filled_value": "5000.00",
                        "average_filled_price": "50000.00",
                        "created_time": "2024-01-01T00:00:00Z",
                    }
                ],
                "cursor": "next_page",
            },
            {
                "orders": [
                    {
                        "order_id": "order_2",
                        "product_id": "BTC-USD",
                        "side": "SELL",
                        "status": "OPEN",
                        "order_type": "LIMIT",
                        "base_size": "0.2",
                        "filled_size": "0",
                        "filled_value": "0",
                        "created_time": "2024-01-01T01:00:00Z",
                    }
                ],
                "cursor": None,
            },
        ]

        result = self.service.list_orders()

        assert len(result) == 2
        assert self.client.list_orders.call_count == 2

    def test_list_orders_exception_raises_query_error(self) -> None:
        """Test list orders raises OrderQueryError on exception."""
        self.client.list_orders.side_effect = Exception("API error")

        with pytest.raises(OrderQueryError, match="Failed to list orders"):
            self.service.list_orders()

    def test_get_order_returns_order(self) -> None:
        """Test getting a single order."""
        self.client.get_order_historical.return_value = {
            "order": {
                "order_id": "order_123",
                "product_id": "BTC-USD",
                "side": "BUY",
                "status": "FILLED",
                "order_type": "LIMIT",
                "base_size": "0.1",
                "filled_size": "0.1",
                "filled_value": "5000.00",
                "average_filled_price": "50000.00",
                "created_time": "2024-01-01T00:00:00Z",
            }
        }

        result = self.service.get_order("order_123")

        assert result is not None
        assert result.id == "order_123"

    def test_get_order_not_found(self) -> None:
        """Test getting non-existent order."""
        self.client.get_order_historical.return_value = {"order": None}

        result = self.service.get_order("nonexistent")

        assert result is None

    def test_get_order_exception(self) -> None:
        """Test get_order raises OrderQueryError on exception."""
        self.client.get_order_historical.side_effect = Exception("Order not found")

        with pytest.raises(OrderQueryError, match="Failed to get order"):
            self.service.get_order("test_123")

    def test_list_fills_returns_fills(self) -> None:
        """Test listing fills."""
        self.client.list_fills.return_value = {
            "fills": [
                {"fill_id": "fill_1", "price": "50000.00", "size": "0.1"},
                {"fill_id": "fill_2", "price": "50100.00", "size": "0.05"},
            ],
            "cursor": None,
        }

        result = self.service.list_fills(product_id="BTC-USD")

        assert len(result) == 2
        assert result[0]["fill_id"] == "fill_1"

    def test_list_fills_with_pagination(self) -> None:
        """Test listing fills handles pagination."""
        self.client.list_fills.side_effect = [
            {"fills": [{"fill_id": "fill_1"}], "cursor": "next"},
            {"fills": [{"fill_id": "fill_2"}], "cursor": None},
        ]

        result = self.service.list_fills()

        assert len(result) == 2
        assert self.client.list_fills.call_count == 2

    def test_close_position_success(self) -> None:
        """Test successful position close."""
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("0.5")

        self.position_provider.list_positions.return_value = [mock_position]
        self.client.close_position.return_value = {
            "order": {
                "order_id": "close_order",
                "product_id": "BTC-PERP",
                "side": "SELL",
                "status": "PENDING",
                "order_type": "MARKET",
                "base_size": "0.5",
                "filled_size": "0",
                "filled_value": "0",
                "created_time": "2024-01-01T00:00:00Z",
            }
        }

        result = self.service.close_position("BTC-PERP")

        assert result.id == "close_order"
        self.client.close_position.assert_called_once()

    def test_close_position_no_position_raises(self) -> None:
        """Test close_position raises when no position exists."""
        self.position_provider.list_positions.return_value = []

        with pytest.raises(ValidationError, match="No open position"):
            self.service.close_position("BTC-PERP")

    def test_close_position_zero_quantity_raises(self) -> None:
        """Test close_position raises when position has zero quantity."""
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("0")

        self.position_provider.list_positions.return_value = [mock_position]

        with pytest.raises(ValidationError, match="No open position"):
            self.service.close_position("BTC-PERP")

    def test_close_position_uses_fallback_on_error(self) -> None:
        """Test close_position uses fallback on API error."""
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("0.5")

        fallback_order = Mock(spec=Order)
        fallback_order.order_id = "fallback_order"

        self.position_provider.list_positions.return_value = [mock_position]
        self.client.close_position.side_effect = Exception("API error")

        result = self.service.close_position("BTC-PERP", fallback=lambda: fallback_order)

        assert result == fallback_order

    def test_close_position_raises_without_fallback(self) -> None:
        """Test close_position raises when no fallback provided."""
        mock_position = Mock(spec=Position)
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("0.5")

        self.position_provider.list_positions.return_value = [mock_position]
        self.client.close_position.side_effect = Exception("API error")

        with pytest.raises(Exception, match="API error"):
            self.service.close_position("BTC-PERP")
