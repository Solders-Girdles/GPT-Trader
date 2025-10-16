from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.features.brokerages.core.interfaces import Position
from bot_v2.features.live_trade.session.account import (
    get_account,
    get_account_snapshot,
    get_market_hours,
    get_orders,
    get_positions,
    get_positions_trading,
    get_quote,
)
from bot_v2.features.live_trade.types import AccountInfo


def _make_position(symbol: str, side: str) -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal("1"),
        entry_price=Decimal("25000"),
        mark_price=Decimal("25500"),
        unrealized_pnl=Decimal("500"),
        realized_pnl=Decimal("0"),
        leverage=None,
        side=side,
    )


def test_get_account_success(live_trade_context, monkeypatch):
    handler = live_trade_context.install_account()
    live_trade_context.set_account(live_trade_context.broker.account)

    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    account = get_account()

    assert isinstance(account, AccountInfo)
    assert handler.calls


def test_get_account_network_error(monkeypatch):
    monkeypatch.setattr(
        "bot_v2.features.live_trade.session.account.get_broker_client",
        lambda: None,
    )
    monkeypatch.setattr("bot_v2.features.live_trade.session.account.log_error", lambda exc: None)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    assert get_account() is None


def test_get_positions_success(live_trade_context, monkeypatch):
    handler = live_trade_context.install_account()
    positions = [_make_position("BTC-USD", "long")]
    live_trade_context.set_positions(positions)

    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    retrieved = get_positions()

    assert len(retrieved) == 1
    assert handler.calls


def test_get_positions_network_error(monkeypatch):
    monkeypatch.setattr(
        "bot_v2.features.live_trade.session.account.get_broker_client",
        lambda: None,
    )
    monkeypatch.setattr("bot_v2.features.live_trade.session.account.log_error", lambda exc: None)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    assert get_positions() == []


def test_get_account_snapshot_returns_snapshot(live_trade_context, monkeypatch):
    live_trade_context.install_account()
    live_trade_context.set_account(live_trade_context.broker.account)
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    snapshot = get_account_snapshot()

    assert snapshot is not None
    assert snapshot.account_id == live_trade_context.broker.account.account_id


def test_get_orders_and_quotes(live_trade_context):
    live_trade_context.install_account()
    order = live_trade_context.execution_engine.order_to_return
    live_trade_context.set_orders([order])
    assert get_orders() == [order]
    assert get_quote(order.symbol) is live_trade_context.broker.quote


def test_get_market_hours_when_broker_missing(monkeypatch):
    monkeypatch.setattr(
        "bot_v2.features.live_trade.session.account.get_broker_client",
        lambda: None,
    )

    hours = get_market_hours()
    assert hours.is_open is False


def test_get_positions_trading_converts(live_trade_context, monkeypatch):
    live_trade_context.install_account()
    live_trade_context.set_positions([_make_position("BTC-USD", "long")])
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    trading_positions = get_positions_trading()
    assert trading_positions[0].symbol == "BTC-USD"
