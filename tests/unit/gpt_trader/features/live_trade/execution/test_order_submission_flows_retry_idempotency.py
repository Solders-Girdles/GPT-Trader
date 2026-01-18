"""Order submission flow idempotency tests for OrderSubmitter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

from gpt_trader.core import (
    OrderSide,
    OrderType,
)
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


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
