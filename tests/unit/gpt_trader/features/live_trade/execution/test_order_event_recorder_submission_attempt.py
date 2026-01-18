"""Tests for OrderEventRecorder.record_submission_attempt."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordSubmissionAttempt:
    """Tests for record_submission_attempt method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_submission_attempt_logs_correctly(
        self,
        mock_get_logger: MagicMock,
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that submission attempt is logged correctly."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_submission_attempt(
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
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that submission attempt handles None price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_submission_attempt(
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
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that submission attempt handles exceptions gracefully."""
        mock_logger = MagicMock()
        mock_logger.log_order_submission.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        order_event_recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )
        mock_logger.log_order_submission.assert_called_once()
