"""Tests for OrderEventRecorder.record_broker_rejection."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordBrokerRejection:
    """Tests for record_broker_rejection method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_broker_rejection_calls_record_rejection(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection calls record_rejection."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_broker_rejection(
            order=order_event_mock_order,
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
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection appends error to event store."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_broker_rejection(
            order=order_event_mock_order,
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
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
    ) -> None:
        """Test that effective_price is used when price is None."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_broker_rejection(
            order=order_event_mock_order,
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
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection handles store exceptions."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_event_store.append_error.side_effect = RuntimeError("Store failure")

        # Should not raise
        order_event_recorder.record_broker_rejection(
            order=order_event_mock_order,
            status_name="REJECTED",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
        )
        mock_emit_metric.assert_called_once()
        mock_event_store.append_error.assert_called_once()
