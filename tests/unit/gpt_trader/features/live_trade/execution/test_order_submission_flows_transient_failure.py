"""Order submission flow transient failure tests for OrderSubmitter."""

from __future__ import annotations

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
from gpt_trader.utilities.datetime_helpers import utc_now


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
                submitted_at=utc_now(),
                updated_at=utc_now(),
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
