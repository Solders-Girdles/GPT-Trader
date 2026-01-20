"""Core unit tests for OrderSubmitter order result handling."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


@pytest.fixture
def monitoring_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    return mock_logger


@pytest.fixture
def emit_metric_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_emit = MagicMock()
    monkeypatch.setattr(recorder_module, "emit_metric", mock_emit)
    return mock_emit


class TestHandleOrderResult:
    """Tests for _handle_order_result method."""

    def test_successful_order_tracked(
        self,
        submitter: OrderSubmitter,
        mock_order: Order,
        open_orders: list[str],
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that successful orders are tracked."""
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

    def test_rejected_status_raises_error(
        self,
        submitter: OrderSubmitter,
        rejected_order: MagicMock,
        emit_metric_mock: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that rejected status raises RuntimeError."""
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

    def test_integration_mode_returns_order_object(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that integration mode returns the order object."""
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
