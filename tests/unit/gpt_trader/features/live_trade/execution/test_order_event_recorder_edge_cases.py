"""Edge-case tests for OrderEventRecorder."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as order_event_recorder_module
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


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
        monkeypatch.setattr(
            order_event_recorder_module, "get_monitoring_logger", lambda: mock_logger
        )
        monkeypatch.setattr(order_event_recorder_module, "emit_metric", mock_emit_metric)

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
        from gpt_trader.core import OrderSide

        mock_logger = MagicMock()
        monkeypatch.setattr(
            order_event_recorder_module, "get_monitoring_logger", lambda: mock_logger
        )

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
