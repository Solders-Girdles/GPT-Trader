"""Order submission flow tests for OrderSubmitter."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.persistence.orders_store import (
    OrdersStore,
)
from gpt_trader.persistence.orders_store import (
    OrderStatus as StoreOrderStatus,
)


class TestSubmitOrder:
    """Tests for submit_order method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_order_submission(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
        mock_order: Order,
        open_orders: list[str],
    ) -> None:
        """Test successful order submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        result = submitter.submit_order(
            symbol="BTC-PERP",
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

        assert result == "order-123"
        assert "order-123" in open_orders

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejected_order_returns_none(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that rejected orders return None (error is caught internally)."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rejected_order = MagicMock()
        rejected_order.id = "order-rejected"
        rejected_order.status = OrderStatus.CANCELLED
        mock_broker.place_order.return_value = rejected_order

        result = submitter.submit_order(
            symbol="BTC-PERP",
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

        assert result is None
        assert "order-rejected" not in open_orders

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_broker_exception_returns_none(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that broker exceptions result in None return."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.side_effect = RuntimeError("API error")

        result = submitter.submit_order(
            symbol="BTC-PERP",
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

        assert result is None

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_order_with_none_result_returns_none(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that None order result returns None."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = None

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
            client_order_id="test-id",
        )

        assert result is None

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_order_with_none_result_persists_terminal_status(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        tmp_path,
    ) -> None:
        """Test that None order result persists a terminal status."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = None

        orders_store = OrdersStore(tmp_path)
        orders_store.initialize()
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            orders_store=orders_store,
        )

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
            client_order_id="test-id",
        )

        assert result is None
        assert orders_store.get_pending_orders() == []
        record = orders_store.get_order("test-id")
        assert record is not None
        assert record.status in {StoreOrderStatus.REJECTED, StoreOrderStatus.FAILED}


class TestOrderSubmissionIntegration:
    """Integration tests for order submission workflows."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_full_order_submission_flow(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test complete order submission flow."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            client_order_id="custom-id",
        )

        assert result == "order-123"
        assert "order-123" in open_orders
        mock_broker.place_order.assert_called_once()
        mock_event_store.append_trade.assert_called_once()

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejection_flow(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test order rejection flow."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rejected_order = MagicMock()
        rejected_order.id = "rejected-order"
        rejected_order.status = MagicMock()
        rejected_order.status.value = "FAILED"
        rejected_order.quantity = Decimal("1.0")
        rejected_order.filled_quantity = Decimal("0")
        mock_broker.place_order.return_value = rejected_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        result = submitter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
            client_order_id=None,
        )

        assert result is None
        assert "rejected-order" not in open_orders
        mock_emit_metric.assert_called()


class TestRetryPathIdempotency:
    """Tests ensuring retry paths don't create duplicate client_order_ids."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_provided_client_order_id_is_reused_on_retry(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that a provided client_order_id is reused across retries."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        captured_client_ids: list[str] = []

        def capture_client_id(**kwargs):
            captured_client_ids.append(kwargs.get("client_id", ""))
            raise RuntimeError("Simulated transient error")

        mock_broker.place_order.side_effect = capture_client_id

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
            client_order_id="retry-test-123",
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
            client_order_id="retry-test-123",
        )

        assert len(captured_client_ids) == 2
        assert captured_client_ids[0] == "retry-test-123"
        assert captured_client_ids[1] == "retry-test-123"
        assert captured_client_ids[0] == captured_client_ids[1]

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_generated_client_order_id_differs_per_submission(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that auto-generated client_order_ids are unique per submission."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        captured_client_ids: list[str] = []

        def capture_client_id(**kwargs):
            captured_client_ids.append(kwargs.get("client_id", ""))
            raise RuntimeError("Simulated error")

        mock_broker.place_order.side_effect = capture_client_id

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

        assert len(captured_client_ids) == 2
        assert captured_client_ids[0] != captured_client_ids[1]
        assert captured_client_ids[0].startswith("test-bot_")
        assert captured_client_ids[1].startswith("test-bot_")


class TestTransientFailureWithClientOrderIdReuse:
    """Integration test for transient failure followed by success."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_transient_failure_then_success_reuses_client_order_id(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
    ) -> None:
        """Test that transient failure followed by success reuses client_order_id."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        captured_client_ids: list[str] = []
        call_count = [0]

        def capture_and_respond(**kwargs: Any) -> Order:
            """Capture client_id, fail first, succeed second."""
            captured_client_ids.append(kwargs.get("client_id", ""))
            call_count[0] += 1

            if call_count[0] == 1:
                raise ConnectionError("transient network error")

            return Order(
                id="order-success-123",
                client_id=kwargs.get("client_id", ""),
                symbol=kwargs.get("symbol", "BTC-USD"),
                side=kwargs.get("side", OrderSide.BUY),
                type=kwargs.get("order_type", OrderType.MARKET),
                quantity=kwargs.get("quantity", Decimal("1.0")),
                price=None,
                stop_price=None,
                tif=TimeInForce.GTC,
                status=OrderStatus.PENDING,
                submitted_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

        mock_broker = MagicMock()
        mock_broker.place_order = capture_and_respond
        mock_event_store = MagicMock()

        open_orders: list[str] = []
        fixed_client_id = "idempotent-order-abc123"

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        result1 = submitter.submit_order(
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
            client_order_id=fixed_client_id,
        )

        assert result1 is None
        assert fixed_client_id not in open_orders

        result2 = submitter.submit_order(
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
            client_order_id=fixed_client_id,
        )

        assert result2 is not None
        assert result2 == "order-success-123"

        assert len(captured_client_ids) == 2
        assert captured_client_ids[0] == fixed_client_id
        assert captured_client_ids[1] == fixed_client_id

        assert len(open_orders) == 1
        assert open_orders[0] == "order-success-123"
