"""Tests for OrderEventRecorder submission methods: broker_rejection and submission_attempt."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


@pytest.fixture(autouse=True)
def monitoring_logger(monkeypatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    return mock_logger


@pytest.fixture()
def emit_metric_mock(monkeypatch) -> MagicMock:
    mock_emit_metric = MagicMock()
    monkeypatch.setattr(recorder_module, "emit_metric", mock_emit_metric)
    return mock_emit_metric


class TestRecordBrokerRejection:
    """Tests for record_broker_rejection method."""

    def test_record_broker_rejection_calls_record_rejection(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection calls record_rejection."""
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
        emit_metric_mock.assert_called()
        call_args = emit_metric_mock.call_args
        metric_data = call_args[0][2]
        assert metric_data["reason"] == "broker_status"
        assert metric_data["reason_detail"] == "REJECTED"

    def test_record_broker_rejection_appends_error(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection appends error to event store."""
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

    def test_record_broker_rejection_uses_effective_price_when_price_none(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
    ) -> None:
        """Test that effective_price is used when price is None."""
        order_event_recorder.record_broker_rejection(
            order=order_event_mock_order,
            status_name="REJECTED",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("49500"),
        )

        call_args = emit_metric_mock.call_args
        metric_data = call_args[0][2]
        assert metric_data["price"] == "49500"

    def test_record_broker_rejection_handles_store_exception(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that broker rejection handles store exceptions."""
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
        emit_metric_mock.assert_called_once()
        mock_event_store.append_error.assert_called_once()


class TestRecordSubmissionAttempt:
    """Tests for record_submission_attempt method."""

    def test_record_submission_attempt_logs_correctly(
        self,
        order_event_recorder: OrderEventRecorder,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that submission attempt is logged correctly."""
        order_event_recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
        )

        monitoring_logger.log_order_submission.assert_called_once_with(
            client_order_id="client-123",
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=1.5,
            price=50000.0,
        )

    def test_record_submission_attempt_handles_none_price(
        self,
        order_event_recorder: OrderEventRecorder,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that submission attempt handles None price."""
        order_event_recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
        )

        call_kwargs = monitoring_logger.log_order_submission.call_args.kwargs
        assert call_kwargs["price"] is None

    def test_record_submission_attempt_handles_exception(
        self,
        order_event_recorder: OrderEventRecorder,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that submission attempt handles exceptions gracefully."""
        monitoring_logger.log_order_submission.side_effect = RuntimeError("Log failure")

        # Should not raise
        order_event_recorder.record_submission_attempt(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )
        monitoring_logger.log_order_submission.assert_called_once()
