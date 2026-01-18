"""Order submission metrics recording tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestOrderSubmissionMetrics:
    """Tests for order submission metrics recording."""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics before and after each test."""
        from gpt_trader.monitoring.metrics_collector import reset_all

        reset_all()
        yield
        reset_all()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_order_records_metric(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that successful order records metric with success labels."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

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
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id=None,
        )

        collector = get_metrics_collector()
        success_key = "gpt_trader_order_submission_total{reason=none,result=success,side=buy}"
        assert success_key in collector.counters
        assert collector.counters[success_key] == 1

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_failed_order_records_metric_with_reason(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that failed order records metric with failure reason."""
        from gpt_trader.monitoring.metrics_collector import get_metrics_collector

        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Insufficient balance")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("0.5"),
            price=None,
            effective_price=Decimal("3000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        collector = get_metrics_collector()
        failed_key = (
            "gpt_trader_order_submission_total{reason=insufficient_funds,result=failed,side=sell}"
        )
        assert failed_key in collector.counters
        assert collector.counters[failed_key] == 1


class TestOrderSubmissionMetricLabels:
    """Tests for classification label propagation into metrics."""

    @patch(
        "gpt_trader.features.live_trade.execution.order_submission._record_order_submission_metric"
    )
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_classification_label_used_in_metrics(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_record_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that failure reason classification is used in metrics."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Rate limit exceeded")

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

        mock_record_metric.assert_called()
        call_kwargs = mock_record_metric.call_args[1]
        assert call_kwargs["reason"] == "rate_limit"
        assert call_kwargs["result"] == "failed"
