from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, Position
from bot_v2.features.live_trade.session.actions import (
    cancel_order,
    close_all_positions,
    place_order,
)
from tests.fixtures.live_trade import create_order, default_account


def test_place_order_success(live_trade_context, monkeypatch):
    context = live_trade_context
    context.set_account(default_account())
    expected_order = create_order(order_id="ord-success")
    context.execution_engine.order_to_return = expected_order

    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    order = place_order("BTC-USD", OrderSide.BUY, Decimal("1.5"), order_type=OrderType.MARKET)

    assert order is expected_order
    assert context.execution_engine.calls[0]["symbol"] == "BTC-USD"
    assert context.execution_engine.calls[0]["quantity"] == Decimal("1.5")


def test_place_order_validation_error(live_trade_context, monkeypatch):
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    with pytest.raises(ValidationError):
        place_order("", OrderSide.BUY, 1)


def test_place_order_handles_execution_error(live_trade_context, monkeypatch):
    context = live_trade_context
    context.set_account(default_account())
    context.execution_engine.fail_with = ExecutionError("engine failure")

    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    monkeypatch.setattr("bot_v2.features.live_trade.session.actions.log_error", lambda exc: None)

    result = place_order("BTC-USD", OrderSide.BUY, 1)

    assert result is None


def test_cancel_order_success(live_trade_context, monkeypatch):
    context = live_trade_context
    order = create_order(order_id="ord-123")
    context.set_orders([order])

    monkeypatch.setattr("builtins.print", lambda *a, **k: None)

    assert cancel_order("ord-123") is True


def test_cancel_order_validation_failure(monkeypatch):
    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    monkeypatch.setattr("bot_v2.features.live_trade.session.actions.log_error", lambda exc: None)

    assert cancel_order("") is False


def test_close_all_positions_mixed_results(live_trade_context, monkeypatch):
    context = live_trade_context
    context.set_account(default_account())

    pos_long = Position(
        symbol="BTC-USD",
        quantity=Decimal("1"),
        entry_price=Decimal("20000"),
        mark_price=Decimal("21000"),
        unrealized_pnl=Decimal("1000"),
        realized_pnl=Decimal("0"),
        leverage=None,
        side="long",
    )
    pos_short = Position(
        symbol="ETH-USD",
        quantity=Decimal("2"),
        entry_price=Decimal("1500"),
        mark_price=Decimal("1400"),
        unrealized_pnl=Decimal("-200"),
        realized_pnl=Decimal("0"),
        leverage=None,
        side="short",
    )
    context.set_positions([pos_long, pos_short])

    calls = []
    responses = [create_order(order_id="close-btc"), None]

    def stub_place_order(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    context.execution_engine.place_order = stub_place_order

    monkeypatch.setattr("builtins.print", lambda *a, **k: None)
    monkeypatch.setattr("bot_v2.features.live_trade.session.actions.log_error", lambda exc: None)

    result = close_all_positions()

    assert result is False
    assert len(calls) == 2
