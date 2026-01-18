"""Tests for OrderEventRecorder.record_trade_event."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordTradeEvent:
    """Tests for record_trade_event method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_trade_event_logs_status_change(
        self,
        mock_get_logger: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
    ) -> None:
        """Test that trade event logs status change."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_trade_event(
            order=order_event_mock_order,
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
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that trade event is appended to event store."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_trade_event(
            order=order_event_mock_order,
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
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that trade event handles log exceptions."""
        mock_logger = MagicMock()
        mock_logger.log_order_status_change.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise, and should still try to append trade
        order_event_recorder.record_trade_event(
            order=order_event_mock_order,
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
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that trade event handles event store exceptions."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_event_store.append_trade.side_effect = RuntimeError("Store failure")

        # Should not raise
        order_event_recorder.record_trade_event(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )
        mock_logger.log_order_status_change.assert_called_once()
        mock_event_store.append_trade.assert_called_once()
