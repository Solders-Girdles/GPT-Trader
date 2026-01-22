"""Shared fixtures for `OrderService` tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.brokerages.coinbase.rest.order_service as order_service_module
from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.errors import ValidationError
from gpt_trader.features.brokerages.coinbase.errors import (
    BrokerageError,
    NotFoundError,
    OrderCancellationError,
    OrderQueryError,
)
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService


class OrderServiceTestBase:
    @pytest.fixture
    def mock_client(self) -> MagicMock:
        """Create a mock Coinbase client."""
        return MagicMock()

    @pytest.fixture
    def mock_payload_builder(self) -> MagicMock:
        """Create a mock OrderPayloadBuilder."""
        return MagicMock()

    @pytest.fixture
    def mock_payload_executor(self) -> MagicMock:
        """Create a mock OrderPayloadExecutor."""
        return MagicMock()

    @pytest.fixture
    def mock_position_provider(self) -> MagicMock:
        """Create a mock PositionProvider."""
        provider = MagicMock()
        provider.list_positions.return_value = []
        return provider

    @pytest.fixture
    def order_service(
        self,
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
    def sample_order_response(self) -> dict:
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


def mock_position(symbol: str, quantity: Decimal) -> MagicMock:
    position = MagicMock()
    position.symbol = symbol
    position.quantity = quantity
    return position


def assert_place_order_builds_payload(
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


def assert_place_order_executes_payload(
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


def assert_place_order_passes_all_parameters(
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


def assert_place_order_market_order_includes_price_none(
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


def assert_cancel_order_success(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.cancel_orders.return_value = {
        "results": [{"order_id": "order-123", "success": True}]
    }

    result = order_service.cancel_order("order-123")

    assert result is True
    mock_client.cancel_orders.assert_called_once_with(order_ids=["order-123"])


def assert_cancel_order_raises_for_rejection_or_missing_order(
    order_service: OrderService,
    mock_client: MagicMock,
    response: dict,
    match: str,
) -> None:
    mock_client.cancel_orders.return_value = response

    with pytest.raises(OrderCancellationError, match=match):
        order_service.cancel_order("order-123")


def assert_cancel_order_re_raises_brokerage_error(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.cancel_orders.side_effect = BrokerageError("rate limited")

    with pytest.raises(BrokerageError, match="rate limited"):
        order_service.cancel_order("order-123")


def assert_cancel_order_wraps_unexpected_exception(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.cancel_orders.side_effect = RuntimeError("API error")

    with pytest.raises(OrderCancellationError, match="Unexpected error"):
        order_service.cancel_order("order-123")


def assert_list_orders_returns_orders(
    order_service: OrderService,
    mock_client: MagicMock,
    sample_order_response: dict,
) -> None:
    mock_client.list_orders.return_value = {
        "orders": [sample_order_response],
        "cursor": None,
    }

    result = order_service.list_orders()

    assert len(result) == 1
    assert isinstance(result[0], Order)


def assert_list_orders_empty_response_returns_empty_list(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.list_orders.return_value = {"orders": []}

    result = order_service.list_orders()

    assert result == []


def assert_list_orders_passes_filters_and_limit(
    order_service: OrderService,
    mock_client: MagicMock,
    kwargs: dict,
    expected_call_kwargs: dict,
) -> None:
    mock_client.list_orders.return_value = {"orders": [], "cursor": None}

    order_service.list_orders(**kwargs)

    call_kwargs = mock_client.list_orders.call_args.kwargs
    for key, value in expected_call_kwargs.items():
        assert call_kwargs[key] == value


def assert_list_orders_pagination(
    order_service: OrderService,
    mock_client: MagicMock,
    sample_order_response: dict,
) -> None:
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


def assert_list_orders_handles_exception(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.list_orders.side_effect = RuntimeError("API error")

    with pytest.raises(OrderQueryError, match="Failed to list orders"):
        order_service.list_orders()


def assert_list_orders_re_raises_brokerage_error(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.list_orders.side_effect = BrokerageError("rate limited")

    with pytest.raises(BrokerageError, match="rate limited"):
        order_service.list_orders()


def assert_get_order_returns_order(
    order_service: OrderService,
    mock_client: MagicMock,
    sample_order_response: dict,
) -> None:
    mock_client.get_order_historical.return_value = {"order": sample_order_response}

    result = order_service.get_order("order-123")

    assert isinstance(result, Order)
    mock_client.get_order_historical.assert_called_once_with("order-123")


def assert_get_order_returns_none_when_order_missing(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.get_order_historical.return_value = {"order": None}

    result = order_service.get_order("nonexistent-order")

    assert result is None


def assert_get_order_not_found_error_returns_none(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.get_order_historical.side_effect = NotFoundError("missing")

    result = order_service.get_order("missing")

    assert result is None


def assert_get_order_handles_exception(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.get_order_historical.side_effect = RuntimeError("API error")

    with pytest.raises(OrderQueryError, match="Failed to get order"):
        order_service.get_order("order-123")


def assert_get_order_re_raises_brokerage_error(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.get_order_historical.side_effect = BrokerageError("auth failed")

    with pytest.raises(BrokerageError, match="auth failed"):
        order_service.get_order("order-123")


def assert_list_fills_returns_fills(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
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


def assert_list_fills_empty_response_returns_empty_list(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.list_fills.return_value = {"fills": []}

    result = order_service.list_fills()

    assert result == []


def assert_list_fills_passes_filters(
    order_service: OrderService,
    mock_client: MagicMock,
    kwargs: dict,
    expected_call_kwargs: dict,
) -> None:
    mock_client.list_fills.return_value = {"fills": [], "cursor": None}

    order_service.list_fills(**kwargs)

    call_kwargs = mock_client.list_fills.call_args.kwargs
    for key, value in expected_call_kwargs.items():
        assert call_kwargs[key] == value


def assert_list_fills_pagination(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.list_fills.side_effect = [
        {"fills": [{"fill_id": "fill-1"}], "cursor": "cursor-123"},
        {"fills": [{"fill_id": "fill-2"}], "cursor": None},
    ]

    result = order_service.list_fills()

    assert len(result) == 2
    assert mock_client.list_fills.call_count == 2


def assert_list_fills_handles_exception_logs_and_raises(
    order_service: OrderService,
    mock_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client.list_fills.side_effect = RuntimeError("API error")
    mock_logger = MagicMock()
    monkeypatch.setattr(order_service_module, "logger", mock_logger)

    with pytest.raises(OrderQueryError, match="Failed to list fills"):
        order_service.list_fills()

    mock_logger.error.assert_called_once()


def assert_list_fills_re_raises_brokerage_error(
    order_service: OrderService,
    mock_client: MagicMock,
) -> None:
    mock_client.list_fills.side_effect = BrokerageError("rate limited")

    with pytest.raises(BrokerageError, match="rate limited"):
        order_service.list_fills()


def assert_close_position_success(
    order_service: OrderService,
    mock_client: MagicMock,
    mock_position_provider: MagicMock,
    sample_order_response: dict,
) -> None:
    mock_position_provider.list_positions.return_value = [mock_position("BTC-PERP", Decimal("1.0"))]
    mock_client.close_position.return_value = {"order": sample_order_response}

    result = order_service.close_position("BTC-PERP")

    assert isinstance(result, Order)
    mock_client.close_position.assert_called_once_with({"product_id": "BTC-PERP"})


def assert_close_position_no_open_position_raises(
    order_service: OrderService,
    mock_position_provider: MagicMock,
    positions: list[MagicMock],
) -> None:
    mock_position_provider.list_positions.return_value = positions

    with pytest.raises(ValidationError, match="No open position"):
        order_service.close_position("BTC-PERP")


def assert_close_position_with_client_order_id(
    order_service: OrderService,
    mock_client: MagicMock,
    mock_position_provider: MagicMock,
    sample_order_response: dict,
) -> None:
    mock_position_provider.list_positions.return_value = [mock_position("ETH-PERP", Decimal("2.0"))]
    mock_client.close_position.return_value = {"order": sample_order_response}

    order_service.close_position("ETH-PERP", client_order_id="my-close-123")

    mock_client.close_position.assert_called_once_with(
        {"product_id": "ETH-PERP", "client_order_id": "my-close-123"}
    )


def assert_close_position_fallback_on_exception(
    order_service: OrderService,
    mock_client: MagicMock,
    mock_position_provider: MagicMock,
) -> None:
    mock_position_provider.list_positions.return_value = [mock_position("BTC-PERP", Decimal("1.0"))]
    mock_client.close_position.side_effect = RuntimeError("API error")

    fallback_order = MagicMock()
    fallback = MagicMock(return_value=fallback_order)

    result = order_service.close_position("BTC-PERP", fallback=fallback)

    assert result is fallback_order
    fallback.assert_called_once()


def assert_close_position_no_fallback_raises(
    order_service: OrderService,
    mock_client: MagicMock,
    mock_position_provider: MagicMock,
) -> None:
    mock_position_provider.list_positions.return_value = [mock_position("BTC-PERP", Decimal("1.0"))]
    mock_client.close_position.side_effect = RuntimeError("API error")

    with pytest.raises(RuntimeError, match="API error"):
        order_service.close_position("BTC-PERP")


def assert_close_position_finds_correct_symbol_among_multiple(
    order_service: OrderService,
    mock_client: MagicMock,
    mock_position_provider: MagicMock,
    sample_order_response: dict,
) -> None:
    mock_position_provider.list_positions.return_value = [
        mock_position("ETH-PERP", Decimal("5.0")),
        mock_position("BTC-PERP", Decimal("2.0")),
        mock_position("SOL-PERP", Decimal("10.0")),
    ]
    mock_client.close_position.return_value = {"order": sample_order_response}

    order_service.close_position("BTC-PERP")

    mock_client.close_position.assert_called_once_with({"product_id": "BTC-PERP"})
