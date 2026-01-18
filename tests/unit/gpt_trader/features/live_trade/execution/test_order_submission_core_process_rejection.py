"""Core unit tests for OrderSubmitter rejection processing."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


class TestProcessRejection:
    """Tests for _process_rejection method."""

    def test_integration_mode_stores_event(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        rejected_order: MagicMock,
    ) -> None:
        """Test that integration mode stores rejection event."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        rejected_order.status = MagicMock()
        rejected_order.status.value = "REJECTED"

        result = submitter._process_rejection(
            order=rejected_order,
            status_name="REJECTED",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
            submit_id="test-id",
            store_status=StoreOrderStatus.REJECTED,
        )

        assert result is rejected_order
        mock_event_store.store_event.assert_called_once_with(
            "order_rejected",
            {
                "order_id": "rejected-order",
                "symbol": "BTC-PERP",
                "status": "REJECTED",
            },
        )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_normal_mode_raises_error(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
        rejected_order: MagicMock,
    ) -> None:
        """Test that normal mode raises RuntimeError."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rejected_order.status = MagicMock()
        rejected_order.status.value = "CANCELLED"

        with pytest.raises(RuntimeError, match="CANCELLED"):
            submitter._process_rejection(
                order=rejected_order,
                status_name="CANCELLED",
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=None,
                effective_price=Decimal("50000"),
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=None,
                submit_id="test-id",
                store_status=StoreOrderStatus.CANCELLED,
            )
