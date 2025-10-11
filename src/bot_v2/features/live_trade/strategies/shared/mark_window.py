"""Helpers for maintaining bounded mark/price windows."""

from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from decimal import Decimal


def update_mark_window(
    store: MutableMapping[str, list[Decimal]],
    *,
    symbol: str,
    current_mark: Decimal,
    short_period: int,
    long_period: int,
    recent_marks: Sequence[Decimal] | None = None,
    buffer: int = 5,
) -> list[Decimal]:
    """Update (or seed) the mark window for ``symbol`` and return it."""
    if recent_marks is not None:
        window = [Decimal(str(value)) for value in recent_marks]
    else:
        window = list(store.get(symbol, []))

    window.append(Decimal(str(current_mark)))

    max_window = max(short_period, long_period) + buffer
    if len(window) > max_window:
        window = window[-max_window:]

    store[symbol] = window
    return window


__all__ = ["update_mark_window"]
