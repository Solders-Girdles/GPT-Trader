"""Core unit tests for recording order rejections."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestRecordRejection:
    """Tests for record_rejection method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_logs_and_emits(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that rejection is logged and metric emitted."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["event_type"] == "order_rejected"
        assert call_args[2]["reason"] == "insufficient_funds"
        assert call_args[2]["reason_detail"] == "insufficient_margin"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_includes_client_order_id(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that client_order_id is included in rejection telemetry."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
            client_order_id="custom-order-123",
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["client_order_id"] == "custom-order-123"

        mock_logger.log_order_status_change.assert_called_once()
        log_call_kwargs = mock_logger.log_order_status_change.call_args[1]
        assert log_call_kwargs["client_order_id"] == "custom-order-123"
        assert log_call_kwargs["order_id"] == "custom-order-123"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_without_client_order_id_uses_empty_string(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that missing client_order_id defaults to empty string."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["client_order_id"] == ""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_rejection_with_none_price(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test recording rejection with None price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=None,
            reason="min_notional",
        )

        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["price"] == "market"
