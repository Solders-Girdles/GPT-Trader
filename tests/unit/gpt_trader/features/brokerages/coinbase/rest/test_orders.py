"""Tests for coinbase/rest/orders.py - OrderRestMixin."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.rest.orders import OrderRestMixin
from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)

# ============================================================
# Test helper class that combines mixin with required base methods
# ============================================================


class MockOrderService(OrderRestMixin):
    """Concrete class that uses OrderRestMixin for testing."""

    def __init__(self, client: MagicMock) -> None:
        self.client = client
        self._build_payload_calls: list[dict] = []
        self._execute_payload_calls: list[dict] = []

    def _build_order_payload(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: TimeInForce | str = TimeInForce.GTC,
        client_id: str | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        post_only: bool = False,
        include_client_id: bool = True,
    ) -> dict:
        """Track calls and return a mock payload."""
        payload = {
            "product_id": symbol,
            "side": side.value if hasattr(side, "value") else side,
            "order_type": order_type.value if hasattr(order_type, "value") else order_type,
            "quantity": str(quantity),
        }
        if price is not None:
            payload["price"] = str(price)
        if client_id:
            payload["client_order_id"] = client_id
        self._build_payload_calls.append(payload)
        return payload

    def _execute_order_payload(
        self, symbol: str, payload: dict, client_id: str | None = None
    ) -> Order:
        """Track calls and return a mock order."""
        self._execute_payload_calls.append({"symbol": symbol, "payload": payload})
        return Order(
            id="order-123",
            client_id=client_id or "generated-id",
            symbol=symbol,
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

    def list_positions(self) -> list:
        """Mock list_positions for close_position tests."""
        return getattr(self, "_mock_positions", [])


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_client() -> MagicMock:
    """Create a mock Coinbase client."""
    client = MagicMock()
    return client


@pytest.fixture
def order_service(mock_client: MagicMock) -> MockOrderService:
    """Create a MockOrderService instance."""
    return MockOrderService(client=mock_client)


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
        order_service: MockOrderService,
    ) -> None:
        """Test that place_order builds correct payload."""
        order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
        )

        assert len(order_service._build_payload_calls) == 1
        payload = order_service._build_payload_calls[0]
        assert payload["product_id"] == "BTC-USD"
        assert payload["side"] == "BUY"
        assert payload["quantity"] == "1.5"
        assert payload["price"] == "50000"

    def test_place_order_executes_payload(
        self,
        order_service: MockOrderService,
    ) -> None:
        """Test that place_order executes the built payload."""
        result = order_service.place_order(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
        )

        assert len(order_service._execute_payload_calls) == 1
        assert order_service._execute_payload_calls[0]["symbol"] == "ETH-USD"
        assert isinstance(result, Order)

    def test_place_order_passes_all_parameters(
        self,
        order_service: MockOrderService,
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

        assert len(order_service._build_payload_calls) == 1
        payload = order_service._build_payload_calls[0]
        assert payload["client_order_id"] == "my-order-123"

    def test_place_order_returns_order(
        self,
        order_service: MockOrderService,
    ) -> None:
        """Test that place_order returns an Order object."""
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
        order_service: MockOrderService,
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
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test failed order cancellation."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "order-123", "success": False}]
        }

        result = order_service.cancel_order("order-123")

        assert result is False

    def test_cancel_order_not_found(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test cancellation when order not in results."""
        mock_client.cancel_orders.return_value = {
            "results": [{"order_id": "other-order", "success": True}]
        }

        result = order_service.cancel_order("order-123")

        assert result is False

    def test_cancel_order_empty_results(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test cancellation with empty results."""
        mock_client.cancel_orders.return_value = {"results": []}

        result = order_service.cancel_order("order-123")

        assert result is False

    def test_cancel_order_handles_exception(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions are handled gracefully."""
        mock_client.cancel_orders.side_effect = RuntimeError("API error")

        result = order_service.cancel_order("order-123")

        assert result is False


# ============================================================
# Test: list_orders
# ============================================================


class TestListOrders:
    """Tests for list_orders method."""

    def test_list_orders_returns_orders(
        self,
        order_service: MockOrderService,
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
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by product_id."""
        mock_client.list_orders.return_value = {"orders": [], "cursor": None}

        order_service.list_orders(product_id="BTC-USD")

        call_kwargs = mock_client.list_orders.call_args.kwargs
        assert call_kwargs["product_id"] == "BTC-USD"

    def test_list_orders_with_status_filter(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by status."""
        mock_client.list_orders.return_value = {"orders": [], "cursor": None}

        order_service.list_orders(status=["PENDING", "OPEN"])

        call_kwargs = mock_client.list_orders.call_args.kwargs
        assert call_kwargs["order_status"] == ["PENDING", "OPEN"]

    def test_list_orders_pagination(
        self,
        order_service: MockOrderService,
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
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions stop pagination and return partial results."""
        mock_client.list_orders.side_effect = RuntimeError("API error")

        result = order_service.list_orders()

        assert result == []

    def test_list_orders_respects_limit(
        self,
        order_service: MockOrderService,
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
        order_service: MockOrderService,
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
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that None is returned when order not found."""
        mock_client.get_order_historical.return_value = {"order": None}

        result = order_service.get_order("nonexistent-order")

        assert result is None

    def test_get_order_handles_exception(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions return None."""
        mock_client.get_order_historical.side_effect = RuntimeError("API error")

        result = order_service.get_order("order-123")

        assert result is None


# ============================================================
# Test: list_fills
# ============================================================


class TestListFills:
    """Tests for list_fills method."""

    def test_list_fills_returns_fills(
        self,
        order_service: MockOrderService,
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
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by product_id."""
        mock_client.list_fills.return_value = {"fills": [], "cursor": None}

        order_service.list_fills(product_id="ETH-USD")

        call_kwargs = mock_client.list_fills.call_args.kwargs
        assert call_kwargs["product_id"] == "ETH-USD"

    def test_list_fills_with_order_filter(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test filtering by order_id."""
        mock_client.list_fills.return_value = {"fills": [], "cursor": None}

        order_service.list_fills(order_id="order-456")

        call_kwargs = mock_client.list_fills.call_args.kwargs
        assert call_kwargs["order_id"] == "order-456"

    def test_list_fills_pagination(
        self,
        order_service: MockOrderService,
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
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exceptions stop pagination."""
        mock_client.list_fills.side_effect = RuntimeError("API error")

        result = order_service.list_fills()

        assert result == []


# ============================================================
# Test: close_position
# ============================================================


class TestClosePosition:
    """Tests for close_position method."""

    def test_close_position_success(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test successful position close."""
        # Setup mock position
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        order_service._mock_positions = [mock_position]

        mock_client.close_position.return_value = {"order": sample_order_response}

        result = order_service.close_position("BTC-PERP")

        assert isinstance(result, Order)
        mock_client.close_position.assert_called_once()

    def test_close_position_no_position_raises(
        self,
        order_service: MockOrderService,
    ) -> None:
        """Test that ValidationError is raised when no position exists."""
        order_service._mock_positions = []

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-PERP")

    def test_close_position_zero_quantity_raises(
        self,
        order_service: MockOrderService,
    ) -> None:
        """Test that zero quantity position is not considered open."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("0")
        order_service._mock_positions = [mock_position]

        with pytest.raises(ValidationError, match="No open position"):
            order_service.close_position("BTC-PERP")

    def test_close_position_with_client_order_id(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
        sample_order_response: dict,
    ) -> None:
        """Test passing client_order_id."""
        mock_position = MagicMock()
        mock_position.symbol = "ETH-PERP"
        mock_position.quantity = Decimal("2.0")
        order_service._mock_positions = [mock_position]

        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("ETH-PERP", client_order_id="my-close-123")

        call_args = mock_client.close_position.call_args[0][0]
        assert call_args["client_order_id"] == "my-close-123"

    def test_close_position_fallback_on_exception(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that fallback is called on exception."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        order_service._mock_positions = [mock_position]

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
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test that exception is raised when no fallback provided."""
        mock_position = MagicMock()
        mock_position.symbol = "BTC-PERP"
        mock_position.quantity = Decimal("1.0")
        order_service._mock_positions = [mock_position]

        mock_client.close_position.side_effect = RuntimeError("API error")

        with pytest.raises(RuntimeError, match="API error"):
            order_service.close_position("BTC-PERP")

    def test_close_position_finds_correct_symbol(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
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

        order_service._mock_positions = [pos1, pos2, pos3]

        mock_client.close_position.return_value = {"order": sample_order_response}

        order_service.close_position("BTC-PERP")

        call_args = mock_client.close_position.call_args[0][0]
        assert call_args["product_id"] == "BTC-PERP"


# ============================================================
# Test: Edge cases
# ============================================================


class TestOrderRestMixinEdgeCases:
    """Tests for edge cases."""

    def test_list_orders_empty_response(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test handling of empty orders response."""
        mock_client.list_orders.return_value = {"orders": []}

        result = order_service.list_orders()

        assert result == []

    def test_list_fills_empty_response(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test handling of empty fills response."""
        mock_client.list_fills.return_value = {"fills": []}

        result = order_service.list_fills()

        assert result == []

    def test_cancel_order_missing_results_key(
        self,
        order_service: MockOrderService,
        mock_client: MagicMock,
    ) -> None:
        """Test handling of response without results key."""
        mock_client.cancel_orders.return_value = {}

        result = order_service.cancel_order("order-123")

        assert result is False

    def test_place_order_market_without_price(
        self,
        order_service: MockOrderService,
    ) -> None:
        """Test placing market order without price."""
        result = order_service.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
        )

        assert isinstance(result, Order)
        payload = order_service._build_payload_calls[0]
        assert "price" not in payload
