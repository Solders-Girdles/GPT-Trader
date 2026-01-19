"""Core unit tests for OrderSubmitter failure handling."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestHandleOrderFailure:
    """Tests for _handle_order_failure method."""

    def test_logs_error_and_returns_none(
        self,
        submitter: OrderSubmitter,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that failure is logged and None returned."""
        result = submitter._handle_order_failure(
            exc=RuntimeError("API error"),
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )

        assert result is None
        mock_event_store.append_error.assert_called_once()

    def test_handles_event_store_exception(
        self,
        submitter: OrderSubmitter,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that event store exceptions are suppressed."""
        mock_event_store.append_error.side_effect = RuntimeError("Store error")

        result = submitter._handle_order_failure(
            exc=RuntimeError("API error"),
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )

        assert result is None
