"""Tests for OrderEventRecorder.record_success."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordSuccess:
    """Tests for record_success method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.logger")
    def test_record_success_logs_order_info(
        self,
        mock_logger: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
    ) -> None:
        """Test that successful order is logged."""
        # record_success only logs, doesn't interact with event_store
        # Just verify it doesn't raise
        order_event_recorder.record_success(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            display_price=Decimal("50000"),
            reduce_only=False,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-USD"
        assert call_kwargs["reduce_only"] is False

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.logger")
    def test_record_success_handles_reduce_only(
        self,
        mock_logger: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
    ) -> None:
        """Test that reduce_only flag is handled."""
        order_event_recorder.record_success(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            display_price=Decimal("51000"),
            reduce_only=True,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.logger")
    def test_record_success_handles_market_price(
        self,
        mock_logger: MagicMock,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
    ) -> None:
        """Test that 'market' as display_price is handled."""
        order_event_recorder.record_success(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            display_price="market",
            reduce_only=False,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["price"] == "market"
