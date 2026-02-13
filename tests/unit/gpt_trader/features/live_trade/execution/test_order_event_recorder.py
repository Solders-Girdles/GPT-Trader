"""Tests for OrderEventRecorder: init, decision trace, edge cases, and trade events."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.decision_trace import OrderDecisionTrace
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


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


class TestRecordDecisionTrace:
    """Tests for record_decision_trace method."""

    def test_record_decision_trace_includes_decision_id(
        self,
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        trace = OrderDecisionTrace(
            symbol="BTC-USD",
            side="BUY",
            price=Decimal("50000"),
            equity=Decimal("100000"),
            quantity=Decimal("1.0"),
            reduce_only=False,
            reason="unit_test",
            decision_id="decision-123",
        )

        order_event_recorder.record_decision_trace(trace)

        mock_event_store.append.assert_called_once()
        event_type, payload = mock_event_store.append.call_args[0]
        assert event_type == "order_decision_trace"
        assert payload["decision_id"] == "decision-123"
        assert payload["client_order_id"] == "decision-123"


class TestOrderEventRecorderEdgeCases:
    """Tests for edge cases."""

    def test_recorder_with_empty_bot_id(self, mock_event_store: MagicMock) -> None:
        """Test recorder with empty bot_id."""
        recorder = OrderEventRecorder(event_store=mock_event_store, bot_id="")
        assert recorder._bot_id == ""

    def test_record_rejection_with_decimal_quantity(
        self,
        order_event_recorder: OrderEventRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test rejection with high precision decimal quantity."""
        mock_logger = MagicMock()
        mock_emit_metric = MagicMock()
        monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
        monkeypatch.setattr(recorder_module, "emit_metric", mock_emit_metric)

        order_event_recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.00123456789"),
            price=Decimal("50000.123"),
            reason="test",
        )

        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["quantity"] == "0.00123456789"

    def test_record_trade_event_uses_order_quantity_when_available(
        self,
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that order.quantity is used when available."""
        mock_logger = MagicMock()
        monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)

        order = MagicMock()
        order.id = "order-123"
        order.client_order_id = "client-123"
        order.quantity = Decimal("2.5")  # Different from passed quantity
        order.price = Decimal("50000")
        order.status = "FILLED"

        order_event_recorder.record_trade_event(
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


@pytest.fixture
def monitoring_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    return mock_logger


class TestRecordTradeEvent:
    """Tests for record_trade_event method."""

    def test_record_trade_event_logs_status_change(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that trade event logs status change."""
        order_event_recorder.record_trade_event(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            submit_id="client-123",
        )

        monitoring_logger.log_order_status_change.assert_called_once_with(
            order_id="order-123",
            client_order_id="client-123",
            from_status=None,
            to_status="open",
        )

    def test_record_trade_event_appends_to_event_store(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that trade event is appended to event store."""
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
        assert trade_payload["status"] == "open"

    def test_record_trade_event_handles_log_exception(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that trade event handles log exceptions."""
        monitoring_logger.log_order_status_change.side_effect = RuntimeError("Log failure")

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

        monitoring_logger.log_order_status_change.assert_called_once()
        mock_event_store.append_trade.assert_called_once()

    def test_record_trade_event_handles_store_exception(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that trade event handles event store exceptions."""
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
        monitoring_logger.log_order_status_change.assert_called_once()
        mock_event_store.append_trade.assert_called_once()
