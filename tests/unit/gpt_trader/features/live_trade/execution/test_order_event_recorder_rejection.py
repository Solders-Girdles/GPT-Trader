"""Tests for OrderEventRecorder.record_rejection."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordRejection:
    """Tests for record_rejection method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_emits_metric(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that record_rejection emits metric with correct data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_rejection(
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
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_rejection logs order status change."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_rejection(
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
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_rejection handles None price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_rejection(
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
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_rejection handles logging exceptions."""
        mock_logger = MagicMock()
        mock_logger.log_order_status_change.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        order_event_recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="test_reason",
        )
        mock_emit_metric.assert_called_once()
        mock_logger.log_order_status_change.assert_called_once()
