"""Tests for features/live_trade/execution/order_event_recorder.py."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import (
    OrderSide,
    OrderType,
)
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def recorder(mock_event_store: MagicMock) -> OrderEventRecorder:
    """Create an OrderEventRecorder instance."""
    return OrderEventRecorder(event_store=mock_event_store, bot_id="test-bot-123")


@pytest.fixture
def mock_order() -> MagicMock:
    """Create a mock order object."""
    order = MagicMock()
    order.id = "order-123"
    order.client_order_id = "client-123"
    order.quantity = Decimal("1.0")
    order.price = Decimal("50000")
    order.status = "SUBMITTED"
    return order


# ============================================================
# Test: __init__
# ============================================================


class TestOrderEventRecorderInit:
    """Tests for OrderEventRecorder initialization."""

    def test_init_stores_event_store(self, mock_event_store: MagicMock) -> None:
        """Test that event_store is stored correctly."""
        recorder = OrderEventRecorder(event_store=mock_event_store, bot_id="test-bot")
        assert recorder._event_store is mock_event_store

    def test_init_stores_bot_id(self, mock_event_store: MagicMock) -> None:
        """Test that bot_id is stored correctly."""
        recorder = OrderEventRecorder(event_store=mock_event_store, bot_id="my-bot-id")
        assert recorder._bot_id == "my-bot-id"


# ============================================================
# Test: record_preview
# ============================================================


class TestRecordPreview:
    """Tests for record_preview method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_skips_when_preview_is_none(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_preview does nothing when preview is None."""
        recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview=None,
        )

        mock_emit_metric.assert_not_called()
        mock_get_logger.assert_not_called()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_emits_metric(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that record_preview emits metric with correct data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        preview = {"fee": "0.001", "estimated_fill": "50000"}

        recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
            preview=preview,
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args
        assert call_args[0][0] is mock_event_store
        assert call_args[0][1] == "test-bot-123"
        metric_data = call_args[0][2]
        assert metric_data["event_type"] == "order_preview"
        assert metric_data["symbol"] == "BTC-USD"
        assert metric_data["side"] == "BUY"
        assert metric_data["order_type"] == "LIMIT"
        assert metric_data["quantity"] == "1.5"
        assert metric_data["price"] == "50000"
        assert metric_data["preview"] == preview

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_uses_market_for_none_price(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that 'market' is used when price is None."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            preview={"some": "data"},
        )

        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["price"] == "market"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_logs_event(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_preview logs to monitoring logger."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_preview(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=None,
            preview={"data": "value"},
        )

        mock_logger.log_event.assert_called_once()
        call_kwargs = mock_logger.log_event.call_args.kwargs
        assert call_kwargs["event_type"] == "order_preview"
        assert call_kwargs["symbol"] == "ETH-USD"
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["component"] == "TradingEngine"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_handles_log_exception(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_preview handles logging exceptions gracefully."""
        mock_logger = MagicMock()
        mock_logger.log_event.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"data": "value"},
        )
        mock_emit_metric.assert_called_once()
        mock_logger.log_event.assert_called_once()


# ============================================================
# Test: record_rejection
# ============================================================


class TestRecordRejection:
    """Tests for record_rejection method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_emits_metric(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that record_rejection emits metric with correct data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["event_type"] == "order_rejected"
        assert metric_data["symbol"] == "BTC-USD"
        assert metric_data["reason"] == "insufficient_funds"
        assert metric_data["reason_detail"] == "insufficient_margin"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_logs_status_change(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_rejection logs order status change."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="paused:mark_staleness",
        )

        mock_logger.log_order_status_change.assert_called_once_with(
            order_id="",
            client_order_id="",
            from_status=None,
            to_status="REJECTED",
            reason="paused",
        )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_handles_none_price(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_rejection handles None price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=None,
            reason="mark_staleness",
        )

        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["price"] == "market"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_handles_log_exception(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_rejection handles logging exceptions."""
        mock_logger = MagicMock()
        mock_logger.log_order_status_change.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="test_reason",
        )
        mock_emit_metric.assert_called_once()
        mock_logger.log_order_status_change.assert_called_once()


# ============================================================
# Test: record_submission_attempt
# ============================================================


class TestRecordSubmissionAttempt:
    """Tests for record_submission_attempt method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_submission_attempt_logs_correctly(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that submission attempt is logged correctly."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
        )

        mock_logger.log_order_submission.assert_called_once_with(
            client_order_id="client-123",
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=1.5,
            price=50000.0,
        )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_submission_attempt_handles_none_price(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that submission attempt handles None price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
        )

        call_kwargs = mock_logger.log_order_submission.call_args.kwargs
        assert call_kwargs["price"] is None

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_submission_attempt_handles_exception(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test that submission attempt handles exceptions gracefully."""
        mock_logger = MagicMock()
        mock_logger.log_order_submission.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )
        mock_logger.log_order_submission.assert_called_once()


# ============================================================
# Test: record_success
# ============================================================


class TestRecordSuccess:
    """Tests for record_success method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.logger")
    def test_record_success_logs_order_info(
        self,
        mock_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
    ) -> None:
        """Test that successful order is logged."""
        # record_success only logs, doesn't interact with event_store
        # Just verify it doesn't raise
        recorder.record_success(
            order=mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            display_price=Decimal("50000"),
            reduce_only=False,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-USD"
        assert call_kwargs["reduce_only"] is False

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.logger")
    def test_record_success_handles_reduce_only(
        self,
        mock_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
    ) -> None:
        """Test that reduce_only flag is handled."""
        recorder.record_success(
            order=mock_order,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            display_price=Decimal("51000"),
            reduce_only=True,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.logger")
    def test_record_success_handles_market_price(
        self,
        mock_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
    ) -> None:
        """Test that 'market' as display_price is handled."""
        recorder.record_success(
            order=mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            display_price="market",
            reduce_only=False,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["price"] == "market"


# ============================================================
# Test: record_trade_event
# ============================================================


class TestRecordTradeEvent:
    """Tests for record_trade_event method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_trade_event_logs_status_change(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
    ) -> None:
        """Test that trade event logs status change."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_trade_event(
            order=mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )

        mock_logger.log_order_status_change.assert_called_once_with(
            order_id="order-123",
            client_order_id="client-123",
            from_status=None,
            to_status="SUBMITTED",
        )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_trade_event_appends_to_event_store(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that trade event is appended to event store."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_trade_event(
            order=mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )

        mock_event_store.append_trade.assert_called_once()
        call_args = mock_event_store.append_trade.call_args
        assert call_args[0][0] == "test-bot-123"
        trade_payload = call_args[0][1]
        assert trade_payload["order_id"] == "order-123"
        assert trade_payload["symbol"] == "BTC-USD"
        assert trade_payload["side"] == "BUY"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_trade_event_handles_log_exception(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that trade event handles log exceptions."""
        mock_logger = MagicMock()
        mock_logger.log_order_status_change.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise, and should still try to append trade
        recorder.record_trade_event(
            order=mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )

        mock_logger.log_order_status_change.assert_called_once()
        mock_event_store.append_trade.assert_called_once()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_trade_event_handles_store_exception(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that trade event handles event store exceptions."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_event_store.append_trade.side_effect = RuntimeError("Store failure")

        # Should not raise
        recorder.record_trade_event(
            order=mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )
        mock_logger.log_order_status_change.assert_called_once()
        mock_event_store.append_trade.assert_called_once()


# ============================================================
# Test: record_failure
# ============================================================


class TestRecordFailure:
    """Tests for record_failure method."""

    def test_record_failure_appends_error_to_event_store(
        self,
        recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that failure is recorded to event store."""
        exc = RuntimeError("Order failed")

        recorder.record_failure(
            exc=exc,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )

        mock_event_store.append_error.assert_called_once_with(
            bot_id="test-bot-123",
            message="order_placement_failed",
            context={
                "symbol": "BTC-USD",
                "side": "BUY",
                "quantity": "1.0",
            },
        )

    def test_record_failure_handles_store_exception(
        self,
        recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that record_failure handles store exceptions."""
        mock_event_store.append_error.side_effect = RuntimeError("Store failure")
        exc = RuntimeError("Order failed")

        # Should not raise
        recorder.record_failure(
            exc=exc,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )
        mock_event_store.append_error.assert_called_once()


# ============================================================
# Test: record_broker_rejection
# ============================================================


class TestRecordBrokerRejection:
    """Tests for record_broker_rejection method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_broker_rejection_calls_record_rejection(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection calls record_rejection."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_broker_rejection(
            order=mock_order,
            status_name="REJECTED",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
        )

        # Should emit metric (from record_rejection)
        mock_emit_metric.assert_called()
        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["reason"] == "broker_status"
        assert metric_data["reason_detail"] == "REJECTED"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_broker_rejection_appends_error(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection appends error to event store."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_broker_rejection(
            order=mock_order,
            status_name="CANCELLED",
            symbol="ETH-USD",
            side=OrderSide.SELL,
            quantity=Decimal("2.0"),
            price=None,
            effective_price=Decimal("3000"),
        )

        mock_event_store.append_error.assert_called_once_with(
            bot_id="test-bot-123",
            message="broker_order_rejected",
            context={
                "symbol": "ETH-USD",
                "status": "CANCELLED",
                "quantity": "2.0",
            },
        )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_broker_rejection_uses_effective_price_when_price_none(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
    ) -> None:
        """Test that effective_price is used when price is None."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_broker_rejection(
            order=mock_order,
            status_name="REJECTED",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("49500"),
        )

        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["price"] == "49500"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_broker_rejection_handles_store_exception(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection handles store exceptions."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_event_store.append_error.side_effect = RuntimeError("Store failure")

        # Should not raise
        recorder.record_broker_rejection(
            order=mock_order,
            status_name="REJECTED",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
        )
        mock_emit_metric.assert_called_once()
        mock_event_store.append_error.assert_called_once()


# ============================================================
# Test: record_integration_rejection
# ============================================================


class TestRecordIntegrationRejection:
    """Tests for record_integration_rejection method."""

    def test_record_integration_rejection_stores_event(
        self,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that integration rejection stores event."""
        recorder.record_integration_rejection(
            order=mock_order,
            symbol="BTC-USD",
            status_name="CANCELLED",
        )

        mock_event_store.store_event.assert_called_once_with(
            "order_rejected",
            {
                "order_id": "order-123",
                "symbol": "BTC-USD",
                "status": "CANCELLED",
            },
        )

    def test_record_integration_rejection_with_different_status(
        self,
        recorder: OrderEventRecorder,
        mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that different status names are handled."""
        mock_order.id = "order-456"

        recorder.record_integration_rejection(
            order=mock_order,
            symbol="ETH-USD",
            status_name="FAILED",
        )

        mock_event_store.store_event.assert_called_once_with(
            "order_rejected",
            {
                "order_id": "order-456",
                "symbol": "ETH-USD",
                "status": "FAILED",
            },
        )


# ============================================================
# Test: Edge cases
# ============================================================


class TestOrderEventRecorderEdgeCases:
    """Tests for edge cases."""

    def test_recorder_with_empty_bot_id(self, mock_event_store: MagicMock) -> None:
        """Test recorder with empty bot_id."""
        recorder = OrderEventRecorder(event_store=mock_event_store, bot_id="")
        assert recorder._bot_id == ""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_with_decimal_quantity(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        recorder: OrderEventRecorder,
    ) -> None:
        """Test rejection with high precision decimal quantity."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.00123456789"),
            price=Decimal("50000.123"),
            reason="test",
        )

        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["quantity"] == "0.00123456789"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_trade_event_uses_order_quantity_when_available(
        self,
        mock_get_logger: MagicMock,
        recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that order.quantity is used when available."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order = MagicMock()
        order.id = "order-123"
        order.client_order_id = "client-123"
        order.quantity = Decimal("2.5")  # Different from passed quantity
        order.price = Decimal("50000")
        order.status = "FILLED"

        recorder.record_trade_event(
            order=order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),  # This should be overridden
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )

        call_args = mock_event_store.append_trade.call_args
        trade_payload = call_args[0][1]
        assert trade_payload["quantity"] == "2.5"
