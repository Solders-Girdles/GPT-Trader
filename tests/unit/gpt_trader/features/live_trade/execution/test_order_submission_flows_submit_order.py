"""Order submission flow tests for OrderSubmitter submit_order."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.persistence.orders_store import OrdersStore
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


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
