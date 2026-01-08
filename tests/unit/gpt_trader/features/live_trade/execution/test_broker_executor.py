"""Tests for features/live_trade/execution/broker_executor.py."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_broker() -> MagicMock:
    """Create a mock broker."""
    broker = MagicMock()
    broker.place_order = MagicMock()
    return broker


@pytest.fixture
def executor(mock_broker: MagicMock) -> BrokerExecutor:
    """Create a BrokerExecutor instance."""
    return BrokerExecutor(broker=mock_broker)


@pytest.fixture
def sample_order() -> Order:
    """Create a sample order response."""
    from datetime import datetime

    return Order(
        id="order-123",
        client_id="client-123",
        symbol="BTC-USD",
        side=OrderSide.BUY,
        type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("50000"),
        stop_price=None,
        tif=TimeInForce.GTC,
        status=OrderStatus.PENDING,
        submitted_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


# ============================================================
# Test: __init__
# ============================================================


class TestBrokerExecutorInit:
    """Tests for BrokerExecutor initialization."""

    def test_init_stores_broker(self, mock_broker: MagicMock) -> None:
        """Test that broker is stored correctly."""
        executor = BrokerExecutor(broker=mock_broker)
        assert executor._broker is mock_broker

    def test_init_defaults_integration_mode_false(self, mock_broker: MagicMock) -> None:
        """Test that integration_mode defaults to False."""
        executor = BrokerExecutor(broker=mock_broker)
        assert executor._integration_mode is False

    def test_init_accepts_integration_mode_true(self, mock_broker: MagicMock) -> None:
        """Test that integration_mode can be set to True."""
        executor = BrokerExecutor(broker=mock_broker, integration_mode=True)
        assert executor._integration_mode is True


# ============================================================
# Test: execute_order - Normal execution path
# ============================================================


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


# ============================================================
# Test: Edge cases
# ============================================================


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
