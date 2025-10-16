"""
Live-trade test scaffolding.

Provides broker/execution stubs, order book generators, and helpers to patch the
legacy session modules without duplicating boilerplate in every test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import List, Optional, Tuple
from collections.abc import Callable, Iterable, Sequence

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    TimeInForce,
)
from bot_v2.features.live_trade.types import AccountInfo, MarketHours


def create_order(
    *,
    order_id: str = "ord-001",
    symbol: str = "BTC-USD",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    tif: TimeInForce = TimeInForce.GTC,
    status: OrderStatus = OrderStatus.SUBMITTED,
    quantity: Decimal = Decimal("1"),
    price: Decimal | None = None,
    stop_price: Decimal | None = None,
    client_id: str | None = None,
    submitted_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> Order:
    """Convenience helper for constructing broker Order objects."""

    submitted_at = submitted_at or datetime.utcnow()
    updated_at = updated_at or submitted_at

    return Order(
        id=order_id,
        client_id=client_id,
        symbol=symbol,
        side=side,
        type=order_type,
        tif=tif,
        status=status,
        submitted_at=submitted_at,
        updated_at=updated_at,
        price=price,
        stop_price=stop_price,
        quantity=quantity,
        filled_quantity=Decimal("0"),
        avg_fill_price=None,
    )


def create_position(
    *,
    symbol: str = "BTC-USD",
    quantity: Decimal = Decimal("0.5"),
    entry_price: Decimal = Decimal("25000"),
    mark_price: Decimal = Decimal("25500"),
    unrealized_pnl: Decimal = Decimal("250"),
    realized_pnl: Decimal = Decimal("0"),
    leverage: int | None = None,
) -> Position:
    """Construct a broker Position stub."""

    return Position(
        symbol=symbol,
        quantity=quantity,
        entry_price=entry_price,
        mark_price=mark_price,
        unrealized_pnl=unrealized_pnl,
        realized_pnl=realized_pnl,
        leverage=leverage,
    )


def default_account() -> AccountInfo:
    """Return a baseline AccountInfo stub."""

    return AccountInfo(
        account_id="acct-123",
        cash=50_000.0,
        portfolio_value=55_000.0,
        buying_power=120_000.0,
        positions_value=20_000.0,
        margin_used=10_000.0,
        pattern_day_trader=False,
        day_trades_remaining=3,
        equity=55_000.0,
        last_equity=54_500.0,
    )


class StubExecutionEngine:
    """Simulates the advanced execution engine used in session.actions."""

    def __init__(self) -> None:
        self.fail_with: Exception | None = None
        self.order_to_return: Order | None = create_order()
        self.calls: list[dict] = []

    def place_order(self, **kwargs) -> Order | None:
        self.calls.append(kwargs)
        if self.fail_with:
            raise self.fail_with
        return self.order_to_return


class StubBrokerClient:
    """Simulated broker client covering the subset used by tests."""

    def __init__(self) -> None:
        self.cancel_should_fail = False
        self.positions: list[Position] = []
        self.orders: list[Order] = []
        self.account: AccountInfo = default_account()
        self.quote: Quote | None = Quote(
            symbol="BTC-USD",
            bid=Decimal("25000"),
            ask=Decimal("25010"),
            last=Decimal("25005"),
            ts=datetime.utcnow(),
        )
        self.market_hours = MarketHours(
            is_open=True,
            open_time=datetime.utcnow() - timedelta(hours=1),
            close_time=datetime.utcnow() + timedelta(hours=5),
            extended_hours_open=False,
        )
        self.connected = True
        self.disconnect_calls = 0

    # Session registry expectations
    def connect(self) -> bool:
        return self.connected

    def disconnect(self) -> None:
        self.disconnect_calls += 1

    def get_account_id(self) -> str:
        return self.account.account_id

    # Session account helpers
    def cancel_order(self, order_id: str) -> bool:
        return not self.cancel_should_fail and any(order.id == order_id for order in self.orders)

    def get_positions(self) -> list[Position]:
        return list(self.positions)

    def get_account(self) -> AccountInfo:
        return self.account

    def get_orders(self, _status: str = "open") -> list[Order]:
        return list(self.orders)

    def get_quote(self, symbol: str) -> Quote | None:
        return self.quote if self.quote and self.quote.symbol == symbol else None

    def get_market_hours(self) -> MarketHours:
        return self.market_hours


class DummyErrorHandler:
    """Minimal error handler mirroring the production interface."""

    def __init__(self) -> None:
        self.calls: list[Callable] = []

    def with_retry(self, func: Callable, *, recovery_strategy=None):
        self.calls.append(func)
        return func()


@dataclass
class LiveTradeTestContext:
    """Bundle wiring for live-trade tests."""

    monkeypatch: pytest.MonkeyPatch
    connection: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(is_connected=True))
    execution_engine: StubExecutionEngine = field(default_factory=StubExecutionEngine)
    broker: StubBrokerClient = field(default_factory=StubBrokerClient)
    account: AccountInfo | None = field(default_factory=default_account)
    positions: list[Position] = field(default_factory=list)
    orders: list[Order] = field(default_factory=list)

    def install_actions(self) -> None:
        """Patch session.actions module to use the stubs."""
        from bot_v2.features.live_trade.session import actions as actions_module

        self.monkeypatch.setattr(actions_module, "get_connection", lambda: self.connection)
        self.monkeypatch.setattr(
            actions_module, "get_execution_engine", lambda: self.execution_engine
        )
        self.monkeypatch.setattr(actions_module, "get_broker_client", lambda: self.broker)
        self.monkeypatch.setattr(
            actions_module,
            "get_account",
            lambda: self.account if self.account is not None else None,
        )
        self.monkeypatch.setattr(actions_module, "get_positions", lambda: list(self.positions))

    def install_account(self) -> DummyErrorHandler:
        """Patch session.account module to use the stubs and return the error handler."""
        from bot_v2.features.live_trade.session import account as account_module

        self.monkeypatch.setattr(account_module, "get_broker_client", lambda: self.broker)
        handler = DummyErrorHandler()
        self.monkeypatch.setattr(account_module, "get_error_handler", lambda: handler)
        return handler

    def set_positions(self, positions: Sequence[Position]) -> None:
        self.positions = list(positions)
        self.broker.positions = list(positions)

    def set_orders(self, orders: Sequence[Order]) -> None:
        self.orders = list(orders)
        self.broker.orders = list(orders)

    def set_account(self, account: AccountInfo | None) -> None:
        self.account = account
        if account is not None:
            self.broker.account = account


@pytest.fixture
def live_trade_context(monkeypatch) -> LiveTradeTestContext:
    """Provide a ready-to-use live-trade testing context."""
    context = LiveTradeTestContext(monkeypatch=monkeypatch)
    context.install_actions()
    return context


def build_order_book(
    *,
    mid_price: Decimal = Decimal("100"),
    spread: Decimal = Decimal("0.5"),
    levels: int = 5,
    level_step: Decimal = Decimal("0.25"),
    base_size: Decimal = Decimal("1"),
) -> tuple[list[tuple[Decimal, Decimal]], list[tuple[Decimal, Decimal]]]:
    """
    Create synthetic bid/ask ladders around ``mid_price``.

    Returns (bids, asks) where bids are sorted descending and asks ascending.
    """

    half_spread = spread / 2
    start_bid = mid_price - half_spread
    start_ask = mid_price + half_spread

    bids = [
        (start_bid - (level_step * i), base_size + Decimal(i) * Decimal("0.1"))
        for i in range(levels)
    ]
    asks = [
        (start_ask + (level_step * i), base_size + Decimal(i) * Decimal("0.1"))
        for i in range(levels)
    ]
    return bids, asks


def build_trade_stream(
    *,
    mid_price: Decimal = Decimal("100"),
    count: int = 10,
    step: Decimal = Decimal("0.2"),
    base_size: Decimal = Decimal("0.05"),
) -> list[tuple[datetime, Decimal, Decimal]]:
    """Generate a deterministic trade stream."""

    now = datetime.utcnow()
    return [
        (now - timedelta(seconds=count - i), mid_price + (step * Decimal(i)), base_size)
        for i in range(count)
    ]


__all__ = [
    "LiveTradeTestContext",
    "StubBrokerClient",
    "StubExecutionEngine",
    "build_order_book",
    "build_trade_stream",
    "create_order",
    "create_position",
    "default_account",
    "live_trade_context",
]
