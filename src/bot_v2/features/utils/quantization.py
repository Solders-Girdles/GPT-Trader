from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP


def quantize_size(size: Decimal, step_size: Decimal, min_size: Decimal) -> Decimal:
    """Quantize size to step_size and enforce minimum size.

    - Rounds to the nearest step using HALF_UP
    - Ensures non-zero sizes are at least min_size
    - Returns Decimal('0') if size <= 0
    """
    if size <= 0:
        return Decimal('0')

    if step_size <= 0:
        # Fallback: no step constraint
        quantized = size
    else:
        quantized = (size / step_size).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * step_size

    if Decimal('0') < quantized < min_size:
        quantized = min_size

    return quantized


def quantize_price(price: Decimal, price_increment: Decimal) -> Decimal:
    """Quantize price to the given increment using HALF_UP rounding."""
    if price_increment and price_increment > 0:
        return (price / price_increment).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * price_increment
    return price


def quantize_price_side_aware(price: Decimal, increment: Decimal, side: str) -> Decimal:
    """Quantize price to increment, using floor for buys and ceil for sells."""
    if increment is None or increment == 0:
        return price
    side_normalized = (side or "").lower()
    quotient = price / increment
    if side_normalized == "buy":
        quantized_units = quotient.to_integral_value(rounding=ROUND_DOWN)
    else:
        quantized_units = quotient.to_integral_value(rounding=ROUND_UP)
    return (quantized_units * increment).quantize(increment)
