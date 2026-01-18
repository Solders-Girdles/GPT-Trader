"""Core unit tests for broker order execution via OrderSubmitter."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import Order, OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


class TestExecuteBrokerOrder:
    """Tests for _execute_broker_order method."""

    def test_normal_order_execution(
        self,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
        mock_order: Order,
    ) -> None:
        """Test normal order execution."""
        mock_broker.place_order.return_value = mock_order

        result = submitter._execute_broker_order(
            submit_id="test-id",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=Decimal("49000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        assert result is mock_order
        mock_broker.place_order.assert_called_once()
        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["quantity"] == Decimal("1.0")
        assert call_kwargs["stop_price"] == Decimal("49000")
        assert call_kwargs["tif"] == TimeInForce.GTC

    def test_type_error_propagates(
        self,
        submitter: OrderSubmitter,
        mock_broker: MagicMock,
    ) -> None:
        """Test that TypeError from broker is propagated."""
        mock_broker.place_order.side_effect = TypeError("unexpected keyword argument")

        with pytest.raises(TypeError):
            submitter._execute_broker_order(
                submit_id="test-id",
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                order_quantity=Decimal("1.0"),
                price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=10,
            )
