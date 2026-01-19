"""Tests for BrokerExecutor.execute_order edge cases and error handling."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor


class TestBrokerExecutorEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_execute_order_with_all_none_optional_params(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test execution with all optional params as None."""
        mock_broker.place_order.return_value = sample_order

        result = executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

        assert result is sample_order
        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["price"] is None
        assert call_kwargs["stop_price"] is None
        assert call_kwargs["tif"] is None
        assert call_kwargs["leverage"] is None

    def test_execute_order_with_decimal_precision(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that decimal precision is preserved."""
        mock_broker.place_order.return_value = sample_order

        precise_quantity = Decimal("0.00123456789")
        precise_price = Decimal("50000.123456")

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=precise_quantity,
            price=precise_price,
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["quantity"] == precise_quantity
        assert call_kwargs["price"] == precise_price

    def test_execute_order_propagates_broker_exception(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        """Test that broker exceptions are propagated."""
        mock_broker.place_order.side_effect = RuntimeError("Broker unavailable")

        with pytest.raises(RuntimeError, match="Broker unavailable"):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000"),
                stop_price=None,
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=None,
            )
