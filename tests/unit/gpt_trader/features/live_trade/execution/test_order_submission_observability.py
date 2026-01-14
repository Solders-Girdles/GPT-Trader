"""Order submission observability and classification tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.logging.correlation import correlation_context, get_domain_context
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("rate_limit exceeded", "rate_limit"),
        ("HTTP 429 Too Many Requests", "rate_limit"),
        ("too many requests", "rate_limit"),
        ("Insufficient balance", "insufficient_funds"),
        ("Not enough funds", "insufficient_funds"),
        ("insufficient margin", "insufficient_funds"),
        ("Invalid size", "invalid_size"),
        ("quantity below min_size", "invalid_size"),
        ("amount too small", "invalid_size"),
        ("Invalid price", "invalid_price"),
        ("price tick increment", "invalid_price"),
        ("Request timeout", "timeout"),
        ("Connection timed out", "timeout"),
        ("deadline exceeded", "timeout"),
        ("Connection refused", "network"),
        ("Network error", "network"),
        ("socket closed", "network"),
        ("Order rejected by broker", "broker_rejected"),
        ("Request rejected", "broker_rejected"),
        ("Order failed", "unknown"),
        ("Server error", "unknown"),
        ("Something weird happened", "unknown"),
        ("", "unknown"),
    ],
)
def test_classify_rejection_reason(message: str, expected: str) -> None:
    """Test _classify_rejection_reason helper."""
    from gpt_trader.features.live_trade.execution.order_submission import (
        _classify_rejection_reason,
    )

    assert _classify_rejection_reason(message) == expected


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


class TestCorrelationContextPropagation:
    """Tests for correlation context propagation during order submission."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_order_context_set_during_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that order context is set during order submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        captured_context: dict = {}

        def capture_context(*args, **kwargs):
            captured_context.update(get_domain_context())
            return mock_order

        mock_broker.place_order.side_effect = capture_context

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
            client_order_id="test-order-id",
        )

        assert captured_context.get("order_id") == "test-order-id"
        assert captured_context.get("symbol") == "BTC-USD"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_correlation_context_preserved_during_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that outer correlation context is preserved during submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        captured_context: dict = {}

        def capture_context(*args, **kwargs):
            captured_context.update(get_domain_context())
            return mock_order

        mock_broker.place_order.side_effect = capture_context

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        with correlation_context(cycle=42):
            submitter.submit_order(
                symbol="ETH-USD",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                order_quantity=Decimal("0.5"),
                price=None,
                effective_price=Decimal("3000"),
                stop_price=None,
                tif=None,
                reduce_only=True,
                leverage=None,
                client_order_id=None,
            )

        assert captured_context.get("cycle") == 42
        assert captured_context.get("symbol") == "ETH-USD"
        assert "order_id" in captured_context

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_order_context_cleared_after_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that order context is cleared after submission completes."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        with correlation_context(cycle=1):
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
                client_order_id="test-id",
            )

            context = get_domain_context()
            assert context.get("cycle") == 1
            assert "order_id" not in context
            assert "symbol" not in context


class TestBrokerStatusClassification:
    """Tests for _classify_rejection_reason with broker status strings."""

    def test_broker_rejected_status(self) -> None:
        """Test classification of broker REJECTED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order rejected by broker: REJECTED") == "broker_status"
        assert _classify_rejection_reason("rejected by exchange") == "broker_rejected"

    def test_broker_cancelled_status(self) -> None:
        """Test classification of broker CANCELLED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        result = _classify_rejection_reason("Order rejected by broker: CANCELLED")
        assert result == "broker_status"

    def test_broker_failed_status(self) -> None:
        """Test classification of broker FAILED status."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Order failed") == "unknown"
        assert _classify_rejection_reason("Execution failure") == "unknown"
        assert _classify_rejection_reason("FAILED status") == "unknown"

    def test_timeout_variations(self) -> None:
        """Test various timeout error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Request timeout") == "timeout"
        assert _classify_rejection_reason("Connection timed out") == "timeout"
        assert _classify_rejection_reason("deadline exceeded") == "timeout"
        assert _classify_rejection_reason("context deadline exceeded") == "timeout"

    def test_network_variations(self) -> None:
        """Test various network error messages."""
        from gpt_trader.features.live_trade.execution.order_submission import (
            _classify_rejection_reason,
        )

        assert _classify_rejection_reason("Connection refused") == "network"
        assert _classify_rejection_reason("Network error") == "network"
        assert _classify_rejection_reason("socket closed") == "network"
        assert _classify_rejection_reason("connection reset") == "network"
        assert _classify_rejection_reason("DNS resolution failed") == "network"


class TestRecordRejectionConsistency:
    """Tests for consistent record_rejection calls with reason and client_order_id."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejection_from_broker_status_includes_classified_reason(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that broker rejections use classified reasons in telemetry."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        order = MagicMock()
        order.id = "rejected-order"
        order.quantity = Decimal("1.0")
        order.filled_quantity = Decimal("0")
        order.price = Decimal("50000")
        order.side = OrderSide.BUY
        order.type = OrderType.MARKET
        order.tif = None

        with pytest.raises(RuntimeError, match="Order rejected"):
            submitter._process_rejection(
                order=order,
                status_name="REJECTED",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                effective_price=Decimal("50000"),
                tif=None,
                reduce_only=False,
                leverage=None,
                submit_id="test-id",
                store_status=StoreOrderStatus.REJECTED,
            )

        calls = mock_emit_metric.call_args_list
        rejection_calls = [c for c in calls if c[0][2].get("event_type") == "order_rejected"]
        assert len(rejection_calls) >= 1

        rejection_data = rejection_calls[0][0][2]
        assert rejection_data["reason"] == "broker_status"
        assert rejection_data["reason_detail"] == "REJECTED"

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_exception_rejection_uses_classified_reason(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that exception-based rejections use classified reasons."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("Insufficient balance for order")

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

        mock_event_store.append_error.assert_called_once()
        call_kwargs = mock_event_store.append_error.call_args[1]
        assert call_kwargs["message"] == "order_placement_failed"


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
