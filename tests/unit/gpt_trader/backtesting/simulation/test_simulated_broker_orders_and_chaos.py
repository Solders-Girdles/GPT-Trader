"""Tests for SimulatedBroker order placement, chaos, and cancellation."""

from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class TestSimulatedBrokerPlaceOrder:
    """Tests for place_order method."""

    def test_place_order_requires_quantity(self) -> None:
        broker = SimulatedBroker()

        with pytest.raises(ValueError, match="quantity is required"):
            broker.place_order(
                symbol="BTC-USD",
                side="BUY",
                order_type="MARKET",
                quantity=None,  # type: ignore
            )

    def test_place_limit_order_submitted(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        # Add market data
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        order = broker.place_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("48000"),
        )

        assert order is not None
        assert order.status.value == "SUBMITTED"
        assert order.symbol == "BTC-USD"
        assert order.quantity == Decimal("0.1")


class TestSimulatedBrokerChaos:
    """Tests for ChaosEngine integration in SimulatedBroker."""

    def test_market_order_rejected_by_chaos(self) -> None:
        from datetime import datetime

        from gpt_trader.backtesting.chaos.engine import ChaosEngine
        from gpt_trader.backtesting.types import ChaosScenario
        from gpt_trader.core import OrderSide, OrderStatus, OrderType, Quote

        broker = SimulatedBroker()
        broker._simulation_time = datetime(2024, 1, 1, 12, 0, 0)
        broker._current_quote["BTC-USD"] = Quote(
            symbol="BTC-USD",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last=Decimal("50000"),
            ts=broker._simulation_time,
        )

        chaos = ChaosEngine(broker, seed=1)
        chaos.add_scenario(
            ChaosScenario(
                name="reject_all",
                enabled=True,
                order_error_probability=Decimal("1"),
            )
        )
        chaos.enable()
        broker.set_chaos_engine(chaos)

        order = broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.status == OrderStatus.REJECTED
        assert order.updated_at == broker._simulation_time

    def test_market_order_partial_fill_by_chaos(self) -> None:
        from datetime import datetime

        from gpt_trader.backtesting.chaos.engine import ChaosEngine
        from gpt_trader.backtesting.types import ChaosScenario
        from gpt_trader.core import OrderSide, OrderStatus, OrderType, Quote

        broker = SimulatedBroker()
        broker._simulation_time = datetime(2024, 1, 1, 12, 0, 0)
        broker._current_quote["BTC-USD"] = Quote(
            symbol="BTC-USD",
            bid=Decimal("49900"),
            ask=Decimal("50100"),
            last=Decimal("50000"),
            ts=broker._simulation_time,
        )

        chaos = ChaosEngine(broker, seed=2)
        chaos.add_scenario(
            ChaosScenario(
                name="partial_fill",
                enabled=True,
                partial_fill_probability=Decimal("1"),
                partial_fill_pct=Decimal("50"),
            )
        )
        chaos.enable()
        broker.set_chaos_engine(chaos)

        order = broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
        )

        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("0.5")


class TestSimulatedBrokerCancelOrder:
    """Tests for cancel_order method."""

    def test_cancel_nonexistent_order(self) -> None:
        broker = SimulatedBroker()

        result = broker.cancel_order("nonexistent-order-id")

        assert result is False

    def test_cancel_open_order(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Candle

        broker = SimulatedBroker()
        # Add market data
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50000"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        # Place a limit order
        order = broker.place_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("48000"),
        )

        # Cancel it
        result = broker.cancel_order(order.id)

        assert result is True
        assert order.id not in broker._open_orders
