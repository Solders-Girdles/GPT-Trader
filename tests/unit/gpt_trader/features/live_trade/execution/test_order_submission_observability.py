from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_submission import _classify_rejection_reason
from gpt_trader.logging.correlation import correlation_context, get_domain_context
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus

pytestmark = pytest.mark.usefixtures("monitoring_logger")

PREVIEW_KWARGS = dict(
    symbol="BTC-PERP",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=Decimal("1.0"),
    price=Decimal("50000"),
)
PREVIEW_MARKET_KWARGS = {**PREVIEW_KWARGS, "order_type": OrderType.MARKET, "price": None}
REJECTION_KWARGS = dict(
    symbol="BTC-PERP",
    side="BUY",
    quantity=Decimal("1.0"),
    price=Decimal("50000"),
    reason="insufficient_margin",
)
REJECTION_MARKET_KWARGS = {**REJECTION_KWARGS, "price": None, "reason": "min_notional"}
MARKET_SUBMIT_KWARGS = dict(order_type=OrderType.MARKET, price=None, tif=None, leverage=None)


def _capture_context(mock_broker: MagicMock, mock_order, captured: dict) -> None:
    def capture(*_args, **_kwargs):
        captured.update(get_domain_context())
        return mock_order

    mock_broker.place_order.side_effect = capture


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("rate_limit exceeded", "rate_limit"),
        ("HTTP 429 Too Many Requests", "rate_limit"),
        ("too many requests", "rate_limit"),
        ("Insufficient balance", "insufficient_funds"),
        ("Invalid size", "invalid_size"),
        ("Invalid price", "invalid_price"),
        ("price tick increment", "invalid_price"),
        ("Request timeout", "timeout"),
        ("deadline exceeded", "timeout"),
        ("context deadline exceeded", "timeout"),
        ("Connection refused", "network"),
        ("Network error", "network"),
        ("connection reset", "network"),
        ("DNS resolution failed", "network"),
        ("Order rejected by broker", "broker_rejected"),
        ("Request rejected", "broker_rejected"),
        ("Order rejected by broker: REJECTED", "broker_status"),
        ("Order rejected by broker: CANCELLED", "broker_status"),
        ("rejected by exchange", "broker_rejected"),
        ("Order failed", "unknown"),
        ("FAILED status", "unknown"),
        ("Server error", "unknown"),
        ("Something weird happened", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_rejection_reason(message: str, expected: str) -> None:
    assert _classify_rejection_reason(message) == expected


def test_record_preview_emits_metric(submitter, emit_metric_mock: MagicMock) -> None:
    submitter.record_preview(**PREVIEW_KWARGS, preview={"estimated_fee": "0.1"})

    emit_metric_mock.assert_called_once()
    call_args = emit_metric_mock.call_args[0]
    assert call_args[1] == "test-bot-123"
    assert call_args[2]["event_type"] == "order_preview"


def test_record_preview_skips_none(submitter, emit_metric_mock: MagicMock) -> None:
    submitter.record_preview(**PREVIEW_KWARGS, preview=None)
    emit_metric_mock.assert_not_called()


def test_record_preview_market_price(submitter, emit_metric_mock: MagicMock) -> None:
    submitter.record_preview(**PREVIEW_MARKET_KWARGS, preview={"estimated_fee": "0.1"})
    assert emit_metric_mock.call_args[0][2]["price"] == "market"


def test_record_preview_suppresses_logger_errors(
    submitter,
    emit_metric_mock: MagicMock,
    monitoring_logger: MagicMock,
) -> None:
    monitoring_logger.log_event.side_effect = RuntimeError("Log error")
    submitter.record_preview(**PREVIEW_KWARGS, preview={"estimated_fee": "0.1"})
    emit_metric_mock.assert_called_once()
    monitoring_logger.log_event.assert_called_once()


def test_record_rejection_logs_and_emits(submitter, emit_metric_mock: MagicMock) -> None:
    submitter.record_rejection(**REJECTION_KWARGS)

    call_args = emit_metric_mock.call_args[0]
    assert call_args[2]["event_type"] == "order_rejected"
    assert call_args[2]["reason"] == "insufficient_funds"
    assert call_args[2]["reason_detail"] == "insufficient_margin"


@pytest.mark.parametrize(
    ("client_order_id", "expected"),
    [("custom-order-123", "custom-order-123"), (None, "")],
)
def test_record_rejection_client_order_id(
    client_order_id: str | None,
    expected: str,
    submitter,
    emit_metric_mock: MagicMock,
    monitoring_logger: MagicMock,
) -> None:
    submitter.record_rejection(**REJECTION_KWARGS, client_order_id=client_order_id)
    call_args = emit_metric_mock.call_args[0]
    assert call_args[2]["client_order_id"] == expected
    if client_order_id:
        monitoring_logger.log_order_status_change.assert_called_once()
        log_call_kwargs = monitoring_logger.log_order_status_change.call_args[1]
        assert log_call_kwargs["client_order_id"] == expected
        assert log_call_kwargs["order_id"] == expected


def test_record_rejection_with_none_price(submitter, emit_metric_mock: MagicMock) -> None:
    submitter.record_rejection(**REJECTION_MARKET_KWARGS)
    assert emit_metric_mock.call_args[0][2]["price"] == "market"


def test_rejection_from_broker_status_includes_classified_reason(
    submitter,
    emit_metric_mock: MagicMock,
) -> None:
    order = MagicMock()
    order.id = "rejected-order"
    order.quantity = Decimal("1.0")
    order.filled_quantity = Decimal("0")
    order.price = Decimal("50000")
    order.side = OrderSide.BUY
    order.type = OrderType.MARKET
    order.tif = None
    process_kwargs = dict(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
        price=None,
        effective_price=Decimal("50000"),
        tif=None,
        reduce_only=False,
        leverage=None,
        submit_id="test-id",
    )

    with pytest.raises(RuntimeError, match="Order rejected"):
        submitter._process_rejection(
            order=order,
            status_name="REJECTED",
            store_status=StoreOrderStatus.REJECTED,
            **process_kwargs,
        )

    rejection_data = next(
        call[0][2] for call in emit_metric_mock.call_args_list if call[0][2].get("event_type")
    )
    assert rejection_data["reason"] == "broker_status"
    assert rejection_data["reason_detail"] == "REJECTED"


def test_exception_rejection_uses_classified_reason(
    submitter,
    mock_broker: MagicMock,
    mock_event_store: MagicMock,
    submit_order_call,
) -> None:
    mock_broker.place_order.side_effect = RuntimeError("Insufficient balance for order")
    submit_order_call(submitter, symbol="BTC-USD", **MARKET_SUBMIT_KWARGS)
    call_kwargs = mock_event_store.append_error.call_args[1]
    assert call_kwargs["message"] == "order_placement_failed"


def test_order_context_set_during_submission(
    submitter,
    mock_broker: MagicMock,
    mock_order,
    submit_order_call,
) -> None:
    captured_context: dict = {}
    _capture_context(mock_broker, mock_order, captured_context)
    submit_order_call(submitter, symbol="BTC-USD", client_order_id="test-order-id")
    assert captured_context.get("order_id") == "test-order-id"
    assert captured_context.get("symbol") == "BTC-USD"


def test_correlation_context_preserved_during_submission(
    submitter,
    mock_broker: MagicMock,
    mock_order,
    submit_order_call,
) -> None:
    captured_context: dict = {}
    _capture_context(mock_broker, mock_order, captured_context)
    with correlation_context(cycle=42):
        submit_order_call(
            submitter,
            symbol="ETH-USD",
            side=OrderSide.SELL,
            reduce_only=True,
            **MARKET_SUBMIT_KWARGS,
        )

    assert captured_context.get("cycle") == 42
    assert captured_context.get("symbol") == "ETH-USD"
    assert "order_id" in captured_context


def test_order_context_cleared_after_submission(
    submitter,
    mock_broker: MagicMock,
    mock_order,
    submit_order_call,
) -> None:
    mock_broker.place_order.return_value = mock_order
    with correlation_context(cycle=1):
        submit_order_call(submitter, symbol="BTC-USD", client_order_id="test-id")
        context = get_domain_context()
        assert context.get("cycle") == 1
        assert "order_id" not in context
        assert "symbol" not in context
