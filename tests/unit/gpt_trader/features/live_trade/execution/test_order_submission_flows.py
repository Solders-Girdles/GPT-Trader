"""Order submission flow tests for OrderSubmitter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as order_event_recorder_module
from gpt_trader.core import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestOrderSubmissionFlows:
    """Order submission workflow tests."""

    def test_full_order_submission_flow(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test complete order submission flow."""
        mock_logger = MagicMock()
        monkeypatch.setattr(
            order_event_recorder_module, "get_monitoring_logger", lambda: mock_logger
        )
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

    def test_rejection_flow(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test order rejection flow."""
        mock_logger = MagicMock()
        monkeypatch.setattr(
            order_event_recorder_module, "get_monitoring_logger", lambda: mock_logger
        )
        mock_emit_metric = MagicMock()
        monkeypatch.setattr(order_event_recorder_module, "emit_metric", mock_emit_metric)

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
