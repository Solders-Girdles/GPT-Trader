"""Tests for BrokerExecutor.execute_order normal execution path."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor


class TestExecuteOrderNormal:
    """Tests for execute_order normal execution path."""

    def test_execute_order_calls_broker_with_correct_args(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that execute_order calls broker.place_order with correct arguments."""
        mock_broker.place_order.return_value = sample_order

        result = executor.execute_order(
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

        mock_broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
            client_id="client-123",
        )
        assert result is sample_order

    def test_execute_order_passes_market_order_with_none_price(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that market orders can pass None as price."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
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

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["price"] is None
        assert call_kwargs["tif"] is None

    def test_execute_order_passes_reduce_only_flag(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that reduce_only flag is passed correctly."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.5"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=True,
            leverage=None,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    def test_execute_order_passes_leverage(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that leverage is passed correctly."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=10,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["leverage"] == 10

    def test_execute_order_passes_stop_price(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that stop_price is passed for stop orders."""
        mock_broker.place_order.return_value = sample_order

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("48000"),
            stop_price=Decimal("49000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
        )

        call_kwargs = mock_broker.place_order.call_args.kwargs
        assert call_kwargs["stop_price"] == Decimal("49000")
