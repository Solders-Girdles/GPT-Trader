"""Trailing-stop management helpers."""

from __future__ import annotations

from collections.abc import MutableMapping
from decimal import Decimal


def update_trailing_stop(
    stops: MutableMapping[str, tuple[Decimal, Decimal]],
    *,
    symbol: str,
    side: str,
    current_price: Decimal,
    trailing_pct: Decimal,
) -> bool:
    """
    Update trailing stop state and return True if the stop was triggered.

    ``side`` should be ``"long"`` or ``"short"`` (case-insensitive).
    """
    side = side.lower()
    trailing_pct = Decimal(str(trailing_pct))
    current_price = Decimal(str(current_price))

    if symbol not in stops:
        if side == "long":
            stop_price = current_price * (Decimal("1") - trailing_pct)
        else:
            stop_price = current_price * (Decimal("1") + trailing_pct)
        stops[symbol] = (current_price, stop_price)
        return False

    peak, stop_price = stops[symbol]

    if side == "long":
        if current_price > peak:
            peak = current_price
            stop_price = peak * (Decimal("1") - trailing_pct)
            stops[symbol] = (peak, stop_price)
        return current_price <= stop_price

    if side == "short":
        if current_price < peak:
            peak = current_price
            stop_price = peak * (Decimal("1") + trailing_pct)
            stops[symbol] = (peak, stop_price)
        return current_price >= stop_price

    return False


__all__ = ["update_trailing_stop"]
