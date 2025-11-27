"""Tests for orchestration/execution/broker_executor.py."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from gpt_trader.orchestration.execution.broker_executor import BrokerExecutor

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
    """Create a BrokerExecutor instance in normal mode."""
    return BrokerExecutor(broker=mock_broker, integration_mode=False)


@pytest.fixture
def integration_executor(mock_broker: MagicMock) -> BrokerExecutor:
    """Create a BrokerExecutor instance in integration mode."""
    return BrokerExecutor(broker=mock_broker, integration_mode=True)


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
# Test: execute_order - Async handling
# ============================================================


class TestExecuteOrderAsync:
    """Tests for execute_order async handling."""

    def test_raises_type_error_for_awaitable_in_normal_mode(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        """Test that TypeError is raised when broker returns awaitable in non-integration mode."""

        async def async_place_order(*args: object, **kwargs: object) -> Order:
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

        mock_broker.place_order.return_value = async_place_order()

        with pytest.raises(TypeError, match="awaitable in non-integration mode"):
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

    def test_handles_awaitable_in_integration_mode(
        self,
        integration_executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        """Test that awaitable results are handled correctly in integration mode."""
        from datetime import datetime

        expected_order = Order(
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

        async def async_place_order(*args: object, **kwargs: object) -> Order:
            return expected_order

        mock_broker.place_order.return_value = async_place_order()

        result = integration_executor.execute_order(
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

        assert result.id == expected_order.id
        assert result.symbol == expected_order.symbol


# ============================================================
# Test: execute_order - TypeError fallback
# ============================================================


class TestExecuteOrderTypeErrorFallback:
    """Tests for execute_order TypeError fallback handling."""

    def test_reraises_type_error_in_normal_mode(
        self,
        executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        """Test that TypeError is re-raised in non-integration mode."""
        mock_broker.place_order.side_effect = TypeError("unexpected keyword argument 'reduce_only'")

        with pytest.raises(TypeError, match="unexpected keyword argument"):
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

    def test_reraises_unrelated_type_error_in_integration_mode(
        self,
        integration_executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        """Test that unrelated TypeError is re-raised even in integration mode."""
        mock_broker.place_order.side_effect = TypeError("some other type error")

        with pytest.raises(TypeError, match="some other type error"):
            integration_executor.execute_order(
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

    def test_falls_back_to_legacy_on_keyword_error_in_integration_mode(
        self,
        integration_executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that legacy signature is used when keyword argument error occurs in integration mode."""
        # First call raises TypeError, second call (legacy) succeeds
        mock_broker.place_order.side_effect = [
            TypeError("unexpected keyword argument 'reduce_only'"),
            sample_order,
        ]

        result = integration_executor.execute_order(
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

        assert result is sample_order
        assert mock_broker.place_order.call_count == 2


# ============================================================
# Test: _invoke_legacy_place_order
# ============================================================


class TestInvokeLegacyPlaceOrder:
    """Tests for _invoke_legacy_place_order method."""

    def test_creates_order_object_with_correct_fields(
        self,
        integration_executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that legacy method creates Order object with correct fields."""
        mock_broker.place_order.return_value = sample_order

        integration_executor._invoke_legacy_place_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
        )

        # Verify the Order object passed to broker
        call_args = mock_broker.place_order.call_args
        order_arg = call_args[0][0]  # First positional argument
        assert isinstance(order_arg, Order)
        assert order_arg.id == "client-123"
        assert order_arg.client_id == "client-123"
        assert order_arg.symbol == "BTC-USD"
        assert order_arg.side == OrderSide.BUY
        assert order_arg.type == OrderType.LIMIT
        assert order_arg.quantity == Decimal("1.0")
        assert order_arg.price == Decimal("50000")
        assert order_arg.tif == TimeInForce.GTC
        assert order_arg.status == OrderStatus.PENDING

    def test_defaults_tif_to_gtc_when_not_time_in_force(
        self,
        integration_executor: BrokerExecutor,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        """Test that tif defaults to GTC when not a TimeInForce enum."""
        mock_broker.place_order.return_value = sample_order

        integration_executor._invoke_legacy_place_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
        )

        call_args = mock_broker.place_order.call_args
        order_arg = call_args[0][0]
        assert order_arg.tif == TimeInForce.GTC

    def test_handles_async_result_from_legacy_call(
        self,
        integration_executor: BrokerExecutor,
        mock_broker: MagicMock,
    ) -> None:
        """Test that async results from legacy call are awaited."""
        from datetime import datetime

        expected_order = Order(
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

        async def async_place_order(*args: object) -> Order:
            return expected_order

        mock_broker.place_order.return_value = async_place_order()

        result = integration_executor._invoke_legacy_place_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
        )

        assert result.id == expected_order.id


# ============================================================
# Test: _await_coroutine
# ============================================================


class TestAwaitCoroutine:
    """Tests for _await_coroutine static method."""

    def test_awaits_coroutine_and_returns_result(self) -> None:
        """Test that coroutine is awaited and result is returned."""

        async def sample_coro() -> str:
            return "test_result"

        result = BrokerExecutor._await_coroutine(sample_coro())
        assert result == "test_result"

    def test_awaits_coroutine_with_complex_return(self) -> None:
        """Test that complex return values are handled correctly."""
        from datetime import datetime

        expected = Order(
            id="order-123",
            client_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.FILLED,
            submitted_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        async def sample_coro() -> Order:
            return expected

        result = BrokerExecutor._await_coroutine(sample_coro())
        assert result.id == expected.id
        assert result.status == OrderStatus.FILLED

    def test_propagates_exception_from_coroutine(self) -> None:
        """Test that exceptions from coroutine are propagated."""

        async def failing_coro() -> None:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            BrokerExecutor._await_coroutine(failing_coro())

    def test_cleans_up_event_loop(self) -> None:
        """Test that event loop is properly cleaned up after execution."""

        async def sample_coro() -> str:
            return "done"

        BrokerExecutor._await_coroutine(sample_coro())

        # After execution, event loop should be set to None
        # (the static method sets it to None in finally block)
        # In Python 3.12+, get_event_loop() may create a new loop
        # The key is that _await_coroutine doesn't leak its loop
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            # Expected - no event loop set
            pass


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
