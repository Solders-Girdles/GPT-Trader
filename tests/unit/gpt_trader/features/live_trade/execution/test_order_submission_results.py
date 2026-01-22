"""Order submission result handling tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


def _handle_result(submitter: OrderSubmitter, order: Order | None, **overrides):
    payload = {
        "symbol": "BTC-PERP",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("1.0"),
        "price": Decimal("50000"),
        "effective_price": Decimal("50000"),
        "tif": TimeInForce.GTC,
        "reduce_only": False,
        "leverage": 10,
        "submit_id": "test-id",
    }
    payload.update(overrides)
    return submitter._handle_order_result(order=order, **payload)


def _process_rejection(
    submitter: OrderSubmitter,
    order: MagicMock,
    status_name: str,
    store_status: StoreOrderStatus,
    **overrides,
):
    payload = {
        "symbol": "BTC-PERP",
        "side": OrderSide.BUY,
        "order_type": OrderType.LIMIT,
        "quantity": Decimal("1.0"),
        "price": Decimal("50000"),
        "effective_price": Decimal("50000"),
        "tif": TimeInForce.GTC,
        "reduce_only": False,
        "leverage": None,
        "submit_id": "test-id",
    }
    payload.update(overrides)
    return submitter._process_rejection(
        order=order,
        status_name=status_name,
        store_status=store_status,
        **payload,
    )


def test_handle_order_failure_logs_error(
    submitter: OrderSubmitter, mock_event_store: MagicMock
) -> None:
    result = submitter._handle_order_failure(
        exc=RuntimeError("API error"),
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
    )

    assert result is None
    mock_event_store.append_error.assert_called_once()


def test_handle_order_failure_suppresses_event_store_exception(
    submitter: OrderSubmitter,
    mock_event_store: MagicMock,
) -> None:
    mock_event_store.append_error.side_effect = RuntimeError("Store error")

    result = submitter._handle_order_failure(
        exc=RuntimeError("API error"),
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
    )

    assert result is None


def test_handle_order_result_tracks_success(
    submitter: OrderSubmitter,
    mock_order: Order,
    open_orders: list[str],
    monitoring_logger: MagicMock,
) -> None:
    result = _handle_result(submitter, order=mock_order)

    assert result == "order-123"
    assert "order-123" in open_orders


def test_handle_order_result_none_order_returns_none(submitter: OrderSubmitter) -> None:
    result = _handle_result(submitter, order=None, leverage=None)

    assert result is None


def test_handle_order_result_missing_id_returns_none(submitter: OrderSubmitter) -> None:
    order = MagicMock()
    order.id = None

    result = _handle_result(submitter, order=order, leverage=None)

    assert result is None


def test_handle_order_result_rejected_status_raises(
    submitter: OrderSubmitter,
    rejected_order: MagicMock,
    emit_metric_mock: MagicMock,
    monitoring_logger: MagicMock,
) -> None:
    rejected_order.status = MagicMock()
    rejected_order.status.value = "REJECTED"

    with pytest.raises(RuntimeError, match="rejected by broker"):
        _handle_result(submitter, order=rejected_order, leverage=None)


def test_handle_order_result_integration_mode_returns_order(
    mock_broker: MagicMock,
    mock_event_store: MagicMock,
    open_orders: list[str],
    mock_order: Order,
    monitoring_logger: MagicMock,
) -> None:
    submitter = OrderSubmitter(
        broker=mock_broker,
        event_store=mock_event_store,
        bot_id="test-bot",
        open_orders=open_orders,
        integration_mode=True,
    )

    result = _handle_result(submitter, order=mock_order)

    assert result is mock_order


def test_process_rejection_integration_mode_stores_event(
    mock_broker: MagicMock,
    mock_event_store: MagicMock,
    open_orders: list[str],
    rejected_order: MagicMock,
) -> None:
    submitter = OrderSubmitter(
        broker=mock_broker,
        event_store=mock_event_store,
        bot_id="test-bot",
        open_orders=open_orders,
        integration_mode=True,
    )

    rejected_order.status = MagicMock()
    rejected_order.status.value = "REJECTED"

    result = _process_rejection(
        submitter,
        order=rejected_order,
        status_name="REJECTED",
        store_status=StoreOrderStatus.REJECTED,
    )

    assert result is rejected_order
    mock_event_store.store_event.assert_called_once_with(
        "order_rejected",
        {
            "order_id": "rejected-order",
            "symbol": "BTC-PERP",
            "status": "REJECTED",
        },
    )


def test_process_rejection_normal_mode_raises(
    submitter: OrderSubmitter,
    rejected_order: MagicMock,
    emit_metric_mock: MagicMock,
    monitoring_logger: MagicMock,
) -> None:
    rejected_order.status = MagicMock()
    rejected_order.status.value = "CANCELLED"

    with pytest.raises(RuntimeError, match="CANCELLED"):
        _process_rejection(
            submitter,
            order=rejected_order,
            status_name="CANCELLED",
            store_status=StoreOrderStatus.CANCELLED,
            price=None,
            leverage=None,
        )
