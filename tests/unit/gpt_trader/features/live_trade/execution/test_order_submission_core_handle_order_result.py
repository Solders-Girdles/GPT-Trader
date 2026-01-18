"""Core unit tests for OrderSubmitter order result handling."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestHandleOrderResult:
    """Tests for _handle_order_result method."""

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_successful_order_tracked(
        self,
        mock_get_logger: MagicMock,
        submitter: OrderSubmitter,
        mock_order: Order,
        open_orders: list[str],
    ) -> None:
        """Test that successful orders are tracked."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        result = submitter._handle_order_result(
            order=mock_order,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            submit_id="test-id",
        )

        assert result == "order-123"
        assert "order-123" in open_orders

    def test_none_order_returns_none(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that None order returns None."""
        result = submitter._handle_order_result(
            order=None,
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
        )

        assert result is None

    def test_order_without_id_returns_none(
        self,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that order without ID returns None."""
        order = MagicMock()
        order.id = None

        result = submitter._handle_order_result(
            order=order,
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
        )

        assert result is None

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.emit_metric")
    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_rejected_status_raises_error(
        self,
        mock_get_logger: MagicMock,
        mock_emit_metric: MagicMock,
        submitter: OrderSubmitter,
        rejected_order: MagicMock,
    ) -> None:
        """Test that rejected status raises RuntimeError."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        rejected_order.status = MagicMock()
        rejected_order.status.value = "REJECTED"

        with pytest.raises(RuntimeError, match="rejected by broker"):
            submitter._handle_order_result(
                order=rejected_order,
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
            )

    @patch("gpt_trader.features.live_trade.execution.order_event_recorder.get_monitoring_logger")
    def test_integration_mode_returns_order_object(
        self,
        mock_get_logger: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that integration mode returns the order object."""
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        result = submitter._handle_order_result(
            order=mock_order,
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
            submit_id="test-id",
        )

        assert result is mock_order
