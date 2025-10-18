"""Utility builders for liquidity and execution tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Tuple

__all__ = ["build_order_book", "build_trade_stream"]


def build_order_book(
    *,
    mid_price: Decimal = Decimal("100"),
    spread: Decimal = Decimal("0.5"),
    levels: int = 5,
    level_step: Decimal = Decimal("0.25"),
    base_size: Decimal = Decimal("1"),
) -> tuple[list[tuple[Decimal, Decimal]], list[tuple[Decimal, Decimal]]]:
    """Create synthetic bid/ask ladders around ``mid_price``."""

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
