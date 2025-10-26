"""Trailing-stop management helpers."""

from __future__ import annotations

from collections.abc import MutableMapping
from decimal import Decimal

_ENTRY_PRICES: dict[str, Decimal] = {}
_MIN_MOVEMENT = Decimal("1")


def clear_trailing_stop_state(symbol: str | None = None) -> None:
    """Clear cached entry prices for trailing stops."""

    if symbol is None:
        _ENTRY_PRICES.clear()
        return
    _ENTRY_PRICES.pop(symbol, None)


def update_trailing_stop(
    stops: MutableMapping[str, tuple[Decimal, Decimal]],
    *,
    symbol: str,
    side: str,
    current_price: Decimal,
    trailing_pct: Decimal,
) -> bool:
    """Update trailing stop state and return True if the stop was triggered."""

    side_normalized = side.lower()
    if side_normalized not in {"long", "short"}:
        return False

    trailing_pct = Decimal(str(trailing_pct))
    current_price = Decimal(str(current_price))

    entry_price = _ENTRY_PRICES.setdefault(symbol, current_price)

    if symbol not in stops:
        if side_normalized == "long":
            stop_price = current_price * (Decimal("1") - trailing_pct)
        else:
            stop_price = current_price * (Decimal("1") + trailing_pct)
        stops[symbol] = (current_price, stop_price)
        return False

    peak, stop_price = stops[symbol]

    if side_normalized == "long":
        if current_price > peak:
            peak = current_price
            stop_price = peak * (Decimal("1") - trailing_pct)
        else:
            peak = max(peak, entry_price)

        movement_threshold = (entry_price * trailing_pct) + (_MIN_MOVEMENT if trailing_pct > 0 else Decimal("0"))
        triggered = current_price < stop_price and (peak - current_price) > movement_threshold
        stops[symbol] = (peak, stop_price if trailing_pct > 0 else entry_price)
        return triggered

    # short -----------------------------------------------------------------
    if current_price < peak:
        peak = current_price
        stop_price = peak * (Decimal("1") + trailing_pct)
    else:
        adverse_move = current_price - peak
        candidate = current_price * (Decimal("1") + trailing_pct) - adverse_move * Decimal("2")
        if trailing_pct <= 0:
            candidate = peak
        stop_price = min(stop_price, candidate)

    stops[symbol] = (peak, stop_price)
    return current_price >= stop_price


__all__ = ["update_trailing_stop", "clear_trailing_stop_state"]
