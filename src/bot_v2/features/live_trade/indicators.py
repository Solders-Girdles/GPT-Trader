"""Lightweight indicator helpers shared across live trading components."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal

__all__ = [
    "to_decimal",
    "mean_decimal",
    "true_range",
    "relative_strength_index",
]


def to_decimal(value) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def mean_decimal(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return (sum(values, Decimal("0")) / Decimal(len(values))).quantize(
        Decimal("0.00000001"), rounding=ROUND_HALF_UP
    )


def true_range(high: Decimal, low: Decimal, prev_close: Decimal | None) -> Decimal:
    ranges = [high - low]
    if prev_close is not None:
        ranges.append((high - prev_close).copy_abs())
        ranges.append((low - prev_close).copy_abs())
    return max(ranges)


def relative_strength_index(closes: Iterable[Decimal]) -> Decimal:
    closes = list(closes)
    if len(closes) < 2:
        return Decimal("0")
    gains: list[Decimal] = []
    losses: list[Decimal] = []
    prev = closes[0]
    for curr in closes[1:]:
        delta = curr - prev
        if delta > 0:
            gains.append(delta)
            losses.append(Decimal("0"))
        else:
            gains.append(Decimal("0"))
            losses.append(-delta)
        prev = curr
    avg_gain = mean_decimal(gains)
    avg_loss = mean_decimal(losses)
    if avg_loss == 0:
        return Decimal("100")
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return Decimal(rsi).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
