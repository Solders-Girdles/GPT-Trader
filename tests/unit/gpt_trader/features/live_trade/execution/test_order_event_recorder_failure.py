"""Tests for OrderEventRecorder.record_failure."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordFailure:
    """Tests for record_failure method."""

    def test_record_failure_appends_error_to_event_store(
        self,
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that failure is recorded to event store."""
        exc = RuntimeError("Order failed")

        order_event_recorder.record_failure(
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
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that record_failure handles store exceptions."""
        mock_event_store.append_error.side_effect = RuntimeError("Store failure")
        exc = RuntimeError("Order failed")

        # Should not raise
        order_event_recorder.record_failure(
            exc=exc,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )
        mock_event_store.append_error.assert_called_once()
