"""Tests for OrderEventRecorder.record_decision_trace."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.decision_trace import OrderDecisionTrace
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


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
