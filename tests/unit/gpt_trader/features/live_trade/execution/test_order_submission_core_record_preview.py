"""Core unit tests for recording order previews."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestRecordPreview:
    """Tests for record_preview method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_with_preview_data(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test recording a preview with data."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"estimated_fee": "0.1"},
        )

        mock_emit_metric.assert_called_once()
        call_args = mock_emit_metric.call_args[0]
        assert call_args[1] == "test-bot-123"
        assert call_args[2]["event_type"] == "order_preview"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    def test_record_preview_with_none_preview_skips(
        self,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that None preview is skipped."""
        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview=None,
        )

        mock_emit_metric.assert_not_called()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_market_price(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test recording a preview with market price."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            preview={"estimated_fee": "0.1"},
        )

        call_args = mock_emit_metric.call_args[0]
        assert call_args[2]["price"] == "market"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_record_preview_handles_logger_exception(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that logger exceptions are suppressed."""
        mock_logger = MagicMock()
        mock_logger.log_event.side_effect = RuntimeError("Log error")
        mock_get_logger.return_value = mock_logger

        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"estimated_fee": "0.1"},
        )
        mock_emit_metric.assert_called_once()
        mock_logger.log_event.assert_called_once()
