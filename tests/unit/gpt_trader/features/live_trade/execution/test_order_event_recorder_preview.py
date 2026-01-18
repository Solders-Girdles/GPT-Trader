"""Tests for OrderEventRecorder.record_preview."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import (
    OrderSide,
    OrderType,
)
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordPreview:
    """Tests for record_preview method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_skips_when_preview_is_none(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_preview does nothing when preview is None."""
        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview=None,
        )

        mock_emit_metric.assert_not_called()
        mock_get_logger.assert_not_called()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_emits_metric(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that record_preview emits metric with correct data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        preview = {"fee": "0.001", "estimated_fill": "50000"}

        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
            preview=preview,
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args
        assert call_args[0][0] is mock_event_store
        assert call_args[0][1] == "test-bot-123"
        metric_data = call_args[0][2]
        assert metric_data["event_type"] == "order_preview"
        assert metric_data["symbol"] == "BTC-USD"
        assert metric_data["side"] == "BUY"
        assert metric_data["order_type"] == "LIMIT"
        assert metric_data["quantity"] == "1.5"
        assert metric_data["price"] == "50000"
        assert metric_data["preview"] == preview

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_uses_market_for_none_price(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that 'market' is used when price is None."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            preview={"some": "data"},
        )

        call_args = mock_emit_metric.call_args
        metric_data = call_args[0][2]
        assert metric_data["price"] == "market"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_logs_event(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_preview logs to monitoring logger."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        order_event_recorder.record_preview(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=None,
            preview={"data": "value"},
        )

        mock_logger.log_event.assert_called_once()
        call_kwargs = mock_logger.log_event.call_args.kwargs
        assert call_kwargs["event_type"] == "order_preview"
        assert call_kwargs["symbol"] == "ETH-USD"
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["component"] == "TradingEngine"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_handles_log_exception(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        order_event_recorder: OrderEventRecorder,
    ) -> None:
        """Test that record_preview handles logging exceptions gracefully."""
        mock_logger = MagicMock()
        mock_logger.log_event.side_effect = RuntimeError("Log failure")
        mock_get_logger.return_value = mock_logger

        # Should not raise
        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"data": "value"},
        )
        mock_emit_metric.assert_called_once()
        mock_logger.log_event.assert_called_once()
