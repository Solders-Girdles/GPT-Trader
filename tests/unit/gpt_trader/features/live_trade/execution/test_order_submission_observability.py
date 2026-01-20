"""Order submission observability tests for rejection classification and recording."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


@pytest.fixture
def monitoring_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    return mock_logger


@pytest.fixture
def emit_metric_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_emit = MagicMock()
    monkeypatch.setattr(recorder_module, "emit_metric", mock_emit)
    return mock_emit


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("rate_limit exceeded", "rate_limit"),
        ("HTTP 429 Too Many Requests", "rate_limit"),
        ("too many requests", "rate_limit"),
        ("Insufficient balance", "insufficient_funds"),
        ("Not enough funds", "insufficient_funds"),
        ("insufficient margin", "insufficient_funds"),
        ("Invalid size", "invalid_size"),
        ("quantity below min_size", "invalid_size"),
        ("amount too small", "invalid_size"),
        ("Invalid price", "invalid_price"),
        ("price tick increment", "invalid_price"),
        ("Request timeout", "timeout"),
        ("Connection timed out", "timeout"),
        ("deadline exceeded", "timeout"),
        ("Connection refused", "network"),
        ("Network error", "network"),
        ("socket closed", "network"),
        ("Order rejected by broker", "broker_rejected"),
        ("Request rejected", "broker_rejected"),
        ("Order failed", "unknown"),
        ("Server error", "unknown"),
        ("Something weird happened", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_rejection_reason(message: str, expected: str) -> None:
    """Test _classify_rejection_reason helper."""
    from gpt_trader.features.live_trade.execution.order_submission import (
        _classify_rejection_reason,
    )

    assert _classify_rejection_reason(message) == expected


class TestBrokerStatusClassification:
    """Tests for _classify_rejection_reason with broker status strings."""

    def test_broker_rejected_status(self) -> None:
        """Test classification of broker REJECTED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order rejected by broker: REJECTED") == "broker_status"
        assert _classify_rejection_reason("rejected by exchange") == "broker_rejected"

    def test_broker_cancelled_status(self) -> None:
        """Test classification of broker CANCELLED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        result = _classify_rejection_reason("Order rejected by broker: CANCELLED")
        assert result == "broker_status"

    def test_broker_failed_status(self) -> None:
        """Test classification of broker FAILED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order failed") == "unknown"
        assert _classify_rejection_reason("Execution failure") == "unknown"
        assert _classify_rejection_reason("FAILED status") == "unknown"

    def test_timeout_variations(self) -> None:
        """Test various timeout error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Request timeout") == "timeout"
        assert _classify_rejection_reason("Connection timed out") == "timeout"
        assert _classify_rejection_reason("deadline exceeded") == "timeout"
        assert _classify_rejection_reason("context deadline exceeded") == "timeout"

    def test_network_variations(self) -> None:
        """Test various network error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Connection refused") == "network"
        assert _classify_rejection_reason("Network error") == "network"
        assert _classify_rejection_reason("socket closed") == "network"
        assert _classify_rejection_reason("connection reset") == "network"
        assert _classify_rejection_reason("DNS resolution failed") == "network"


class TestRecordRejectionConsistency:
    """Tests for consistent record_rejection calls with reason and client_order_id."""

    def test_rejection_from_broker_status_includes_classified_reason(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        emit_metric_mock: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that broker rejections use classified reasons in telemetry."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        order = MagicMock()
        order.id = "rejected-order"
        order.quantity = Decimal("1.0")
        order.filled_quantity = Decimal("0")
        order.price = Decimal("50000")
        order.side = OrderSide.BUY
        order.type = OrderType.MARKET
        order.tif = None

        with pytest.raises(RuntimeError, match="Order rejected"):
            submitter._process_rejection(
                order=order,
                status_name="REJECTED",
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
                store_status=StoreOrderStatus.REJECTED,
            )

        calls = emit_metric_mock.call_args_list
        rejection_calls = [c for c in calls if c[0][2].get("event_type") == "order_rejected"]
        assert len(rejection_calls) >= 1

        rejection_data = rejection_calls[0][0][2]
        assert rejection_data["reason"] == "broker_status"
        assert rejection_data["reason_detail"] == "REJECTED"

    def test_exception_rejection_uses_classified_reason(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        emit_metric_mock: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that exception-based rejections use classified reasons."""
        mock_broker.place_order.side_effect = RuntimeError("Insufficient balance for order")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        mock_event_store.append_error.assert_called_once()
        call_kwargs = mock_event_store.append_error.call_args[1]
        assert call_kwargs["message"] == "order_placement_failed"
