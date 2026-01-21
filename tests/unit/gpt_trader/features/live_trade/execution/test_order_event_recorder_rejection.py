"""Tests for OrderEventRecorder rejection methods: record_rejection, integration_rejection, failure."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide
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


class TestRecordRejection:
    """Tests for record_rejection method."""

    def test_record_rejection_emits_metric(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that record_rejection emits metric with correct data."""
        order_event_recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        emit_metric_mock.assert_called_once()
        call_args = emit_metric_mock.call_args
        metric_data = call_args[0][2]
        assert metric_data["event_type"] == "order_rejected"
        assert metric_data["symbol"] == "BTC-USD"
        assert metric_data["reason"] == "insufficient_funds"
        assert metric_data["reason_detail"] == "insufficient_margin"

    def test_record_rejection_logs_status_change(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that record_rejection logs order status change."""
        order_event_recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="paused:mark_staleness",
        )

        monitoring_logger.log_order_status_change.assert_called_once_with(
            order_id="",
            client_order_id="",
            from_status=None,
            to_status="REJECTED",
            reason="paused",
        )

    def test_record_rejection_handles_none_price(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_rejection handles None price."""
        order_event_recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=None,
            reason="mark_staleness",
        )

        call_args = emit_metric_mock.call_args
        metric_data = call_args[0][2]
        assert metric_data["price"] == "market"

    def test_record_rejection_handles_log_exception(
        self,
        emit_metric_mock: MagicMock,
        order_event_recorder: OrderEventRecorder,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that record_rejection handles logging exceptions."""
        monitoring_logger.log_order_status_change.side_effect = RuntimeError("Log failure")

        # Should not raise
        order_event_recorder.record_rejection(
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="test_reason",
        )
        emit_metric_mock.assert_called_once()
        monitoring_logger.log_order_status_change.assert_called_once()


class TestRecordIntegrationRejection:
    """Tests for record_integration_rejection method."""

    def test_record_integration_rejection_stores_event(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that integration rejection stores event."""
        order_event_recorder.record_integration_rejection(
            order=order_event_mock_order,
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
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that different status names are handled."""
        order_event_mock_order.id = "order-456"

        order_event_recorder.record_integration_rejection(
            order=order_event_mock_order,
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
