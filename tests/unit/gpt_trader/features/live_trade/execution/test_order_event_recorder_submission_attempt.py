"""Tests for OrderEventRecorder.record_submission_attempt."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as order_event_recorder_module
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordSubmissionAttempt:
    """Tests for record_submission_attempt method."""

    def test_record_submission_attempt_logs_correctly(
        self,
        order_event_recorder: OrderEventRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that submission attempt is logged correctly."""
        mock_logger = MagicMock()
        monkeypatch.setattr(
            order_event_recorder_module, "get_monitoring_logger", lambda: mock_logger
        )

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

    def test_record_submission_attempt_handles_none_price(
        self,
        order_event_recorder: OrderEventRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that submission attempt handles None price."""
        mock_logger = MagicMock()
        monkeypatch.setattr(
            order_event_recorder_module, "get_monitoring_logger", lambda: mock_logger
        )

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

    def test_record_submission_attempt_handles_exception(
        self,
        order_event_recorder: OrderEventRecorder,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that submission attempt handles exceptions gracefully."""
        mock_logger = MagicMock()
        mock_logger.log_order_submission.side_effect = RuntimeError("Log failure")
        monkeypatch.setattr(
            order_event_recorder_module, "get_monitoring_logger", lambda: mock_logger
        )

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
