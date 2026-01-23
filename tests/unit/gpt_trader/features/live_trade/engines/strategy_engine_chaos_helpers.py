"""Helper utilities for TradingEngine chaos tests."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.core import Position


def make_position(symbol: str = "BTC-USD", qty: str = "1.0", side: str = "long") -> Position:
    return Position(
        symbol=symbol,
        quantity=Decimal(qty),
        entry_price=Decimal("40000"),
        mark_price=Decimal("50000"),
        unrealized_pnl=Decimal("10000"),
        realized_pnl=Decimal("0"),
        side=side,
    )
