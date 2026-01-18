"""Order submission latency metrics tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import Order, OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestOrderSubmissionLatencyMetrics:
    """Tests for order submission latency metrics recording."""

    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_latency"
    )
    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_metric"
    )
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_submission_records_latency_histogram(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_record_metric: MagicMock,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that successful submission records latency histogram."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["result"] == "success"
        assert call_kwargs["side"].lower() == "buy"
        assert call_kwargs["latency_seconds"] >= 0

    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_latency"
    )
    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_metric"
    )
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_failed_submission_records_latency_with_failure_result(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_record_metric: MagicMock,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that failed submission records latency with failure result."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Connection error")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["result"] == "failed"
        assert call_kwargs["side"].lower() == "sell"
