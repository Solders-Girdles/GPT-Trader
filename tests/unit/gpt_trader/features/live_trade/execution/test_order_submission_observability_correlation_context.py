"""Order submission correlation context propagation tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.logging.correlation import correlation_context, get_domain_context


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
