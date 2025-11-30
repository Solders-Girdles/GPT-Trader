"""Tests for coinbase/rest/order_service.py - OrderService."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.errors import (
    OrderCancellationError,
    OrderQueryError,
)
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock Coinbase client."""
    client = MagicMock()
    return client


@pytest.fixture
def mock_payload_builder() -> MagicMock:
    """Create a mock OrderPayloadBuilder."""
    builder = MagicMock()
    return builder


@pytest.fixture
def mock_payload_executor() -> MagicMock:
    """Create a mock OrderPayloadExecutor."""
    executor = MagicMock()
    return executor


@pytest.fixture
def mock_position_provider() -> MagicMock:
    """Create a mock PositionProvider."""
    provider = MagicMock()
    # Default list_positions to empty list
    provider.list_positions.return_value = []
    return provider


@pytest.fixture
def order_service(
    mock_client: MagicMock,
    mock_payload_builder: MagicMock,
    mock_payload_executor: MagicMock,
    mock_position_provider: MagicMock,
) -> OrderService:
    """Create an OrderService instance with mocked dependencies."""
    return OrderService(
        client=mock_client,
        payload_builder=mock_payload_builder,
        payload_executor=mock_payload_executor,
        position_provider=mock_position_provider,
    )


@pytest.fixture
def sample_order_response() -> dict:
    """Create a sample order response from the API."""
    return {
        "order_id": "order-123",
        "client_order_id": "client-123",
        "product_id": "BTC-USD",
        "side": "BUY",
        "order_type": "LIMIT",
        "base_size": "1.0",
        "limit_price": "50000",
        "status": "PENDING",
        "created_time": "2024-01-01T00:00:00Z",
        "last_fill_time": None,
    }


# ============================================================
# Test: place_order
# ============================================================


class TestPlaceOrder:
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


# ============================================================
# Test: cancel_order
# ============================================================


class TestCancelOrder:
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


# ============================================================
# Test: list_orders
# ============================================================


class TestListOrders:
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
        # First page with cursor
        page1_response = sample_order_response.copy()
        page1_response["order_id"] = "order-1"
        # Second page without cursor
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


# ============================================================
# Test: get_order
# ============================================================


class TestGetOrder:
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


# ============================================================
# Test: list_fills
# ============================================================


class TestListFills:
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


# ============================================================
# Test: close_position
# ============================================================


class TestClosePosition:
    """Tests for close_position method."""

    def test_close_position_success(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test successful position close."""
        # Setup mock position
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.return_value = {"order": sample_order_response}

        result = order_service.close_position("BTC-PERP")

        assert isinstance(result, Order)
        mock_client.close_position.assert_called_once()

    def test_close_position_no_position_raises(
        self,
        order_service: OrderService,
        mock_position_provider: MagicMock,
    ) -> None:
        """Test that ValidationError is raised when no position exists."""
        mock_position_provider.list_positions.return_value = []

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-PERP")

    def test_close_position_zero_quantity_raises(
        self,
        order_service: OrderService,
        mock_position_provider: MagicMock,
    ) -> None:
        """Test that zero quantity position is not considered open."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("0")
        mock_position_provider.list_positions.return_value = [mock_position]

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-PERP")

    def test_close_position_with_client_order_id(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test passing client_order_id."""
        mock_position = MagicMock()
        mock_position.symbol = "ETH-PERP"
        mock_position.quantity = Decimal("2.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("ETH-PERP", client_order_id="my-close-123")

        call_args = mock_client.close_position.call_args[0][0]
        assert call_args["client_order_id"] == "my-close-123"

    def test_close_position_fallback_on_exception(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
    ) -> None:
        """Test that fallback is called on exception."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.side_effect = RuntimeError("API error")

        fallback_order = Order(
            id="fallback-order",
            client_id="fallback-client",
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.PENDING,
            submitted_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        fallback = MagicMock(return_value=fallback_order)

        result = order_service.close_position("BTC-PERP", fallback=fallback)

        assert result is fallback_order
        fallback.assert_called_once()

    def test_close_position_no_fallback_raises(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
    ) -> None:
        """Test that exception is raised when no fallback provided."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        mock_position_provider.list_positions.return_value = [mock_position]

        mock_client.close_position.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            order_service.close_position("BTC-PERP")

    def test_close_position_finds_correct_symbol(
        self,
        order_service: OrderService,
        mock_client: MagicMock,
        mock_position_provider: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test that correct symbol position is found among multiple."""
        pos1 = MagicMock()
        pos1.symbol = "ETH-PERP"
        pos1.quantity = Decimal("5.0")

        pos2 = MagicMock()
        pos2.symbol = "BTC-PERP"
        pos2.quantity = Decimal("2.0")

        pos3 = MagicMock()
        pos3.symbol = "SOL-PERP"
        pos3.quantity = Decimal("10.0")

        mock_position_provider.list_positions.return_value = [pos1, pos2, pos3]

        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("BTC-PERP")

        call_args = mock_client.close_position.call_args[0][0]
        assert call_args["product_id"] == "BTC-PERP"


# ============================================================
# Test: Edge cases
# ============================================================


class TestOrderServiceEdgeCases:
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
        # Setup execute mock to return something
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
