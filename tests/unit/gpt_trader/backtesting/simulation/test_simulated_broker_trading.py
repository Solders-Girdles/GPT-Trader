"""Tests for SimulatedBroker orders, positions, and risk helpers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.backtesting.chaos.engine import ChaosEngine
from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import ChaosScenario
from gpt_trader.core import (
    Candle,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
)

BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)


def make_bar(price: Decimal = Decimal("50000")) -> Candle:
    return Candle(
        ts=BASE_TIME,
        open=price - Decimal("1000"),
        high=price + Decimal("1000"),
        low=price - Decimal("1500"),
        close=price,
        volume=Decimal("100"),
    )


def make_quote(price: Decimal = Decimal("50000")) -> Quote:
    return Quote(
        symbol="BTC-USD",
        bid=price - Decimal("10"),
        ask=price + Decimal("10"),
        last=price,
        ts=BASE_TIME,
    )


def make_position(
    quantity: Decimal = Decimal("1.0"),
    entry_price: Decimal = Decimal("50000"),
    mark_price: Decimal = Decimal("50000"),
    side: str = "long",
    leverage: int | None = 5,
    unrealized_pnl: Decimal = Decimal("0"),
    realized_pnl: Decimal = Decimal("0"),
) -> Position:
    return Position(
        symbol="BTC-USD",
        quantity=quantity,
        side=side,
        entry_price=entry_price,
        mark_price=mark_price,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        leverage=leverage,  # type: ignore[arg-type]
    )


class TestSimulatedBrokerOrders:
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
        broker = SimulatedBroker()
        broker._current_bar["BTC-USD"] = make_bar()
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

    def test_cancel_nonexistent_order(self) -> None:
        broker = SimulatedBroker()
        assert broker.cancel_order("nonexistent-order-id") is False

    def test_cancel_open_order(self) -> None:
        broker = SimulatedBroker()
        broker._current_bar["BTC-USD"] = make_bar()
        order = broker.place_order(
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.1"),
            price=Decimal("48000"),
        )

        result = broker.cancel_order(order.id)
        assert result is True
        assert order.id not in broker._open_orders

    def test_cancel_open_order_direct(self) -> None:
        broker = SimulatedBroker()
        order = Order(
            id="test-order-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            status=OrderStatus.SUBMITTED,
            price=Decimal("45000"),
        )
        broker._open_orders[order.id] = order
        result = broker.cancel_order(order.id)
        assert result is True
        assert order.id not in broker._open_orders
        assert broker._cancelled_orders[order.id].status == OrderStatus.CANCELLED


@pytest.mark.parametrize(
    "scenario_kwargs, expected_status, expected_filled",
    [
        (
            {"name": "reject_all", "order_error_probability": Decimal("1")},
            OrderStatus.REJECTED,
            None,
        ),
        (
            {
                "name": "partial_fill",
                "partial_fill_probability": Decimal("1"),
                "partial_fill_pct": Decimal("50"),
            },
            OrderStatus.PARTIALLY_FILLED,
            Decimal("0.5"),
        ),
    ],
)
def test_market_order_chaos_scenarios(
    scenario_kwargs: dict[str, Decimal],
    expected_status: OrderStatus,
    expected_filled: Decimal | None,
) -> None:
    broker = SimulatedBroker()
    broker._simulation_time = BASE_TIME
    broker._current_quote["BTC-USD"] = make_quote()

    chaos = ChaosEngine(broker, seed=1)
    chaos.add_scenario(
        ChaosScenario(name=scenario_kwargs.pop("name"), enabled=True, **scenario_kwargs)
    )
    chaos.enable()
    broker.set_chaos_engine(chaos)
    order = broker.place_order(
        symbol="BTC-USD",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("1.0"),
    )

    assert order.status == expected_status
    if expected_filled is not None:
        assert order.filled_quantity == expected_filled
    if expected_status is OrderStatus.REJECTED:
        assert order.updated_at == broker._simulation_time


@pytest.mark.parametrize(
    "positions, expected",
    [
        ({}, Decimal("0")),
        ({"BTC-USD": make_position()}, Decimal("10000")),
        (
            {
                "BTC-USD": make_position(),
                "ETH-USD": make_position(
                    quantity=Decimal("10.0"),
                    entry_price=Decimal("3000"),
                    mark_price=Decimal("3000"),
                    leverage=3,
                ),
            },
            Decimal("20000"),
        ),
        ({"BTC-USD": make_position(leverage=None)}, Decimal("50000")),
    ],
)
def test_calculate_margin_used(positions: dict[str, Position], expected: Decimal) -> None:
    broker = SimulatedBroker()
    broker.positions = positions
    assert broker._calculate_margin_used() == expected


def test_position_helpers_with_position() -> None:
    broker = SimulatedBroker()
    broker.positions["BTC-USD"] = make_position(
        mark_price=Decimal("55000"),
        unrealized_pnl=Decimal("5000"),
        realized_pnl=Decimal("1000"),
    )
    broker._current_quote["BTC-USD"] = make_quote()
    position = broker.get_position("BTC-USD")
    assert position is not None
    assert position.symbol == "BTC-USD"
    assert position.quantity == Decimal("1.0")

    pnl = broker.get_position_pnl("BTC-USD")
    assert pnl["realized_pnl"] == Decimal("1000")
    assert pnl["unrealized_pnl"] == Decimal("5000")
    assert pnl["total_pnl"] == Decimal("6000")

    risk = broker.get_position_risk("BTC-USD")
    assert risk["symbol"] == "BTC-USD"
    assert risk["notional"] == Decimal("50000")
    assert risk["leverage"] == 5
    assert risk["margin_used"] == Decimal("10000")
    assert "liquidation_price" in risk


def test_position_helpers_without_position() -> None:
    broker = SimulatedBroker()

    assert broker.get_position("BTC-USD") is None
    pnl = broker.get_position_pnl("BTC-USD")
    assert pnl["realized_pnl"] == Decimal("0")
    assert pnl["unrealized_pnl"] == Decimal("0")
    assert pnl["total_pnl"] == Decimal("0")
    assert broker.get_position_risk("BTC-USD") == {}
