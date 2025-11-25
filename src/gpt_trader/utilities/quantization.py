from decimal import ROUND_DOWN, ROUND_UP, Decimal


def quantize_price_side_aware(price: Decimal, increment: Decimal, side: str) -> Decimal:
    """
    Quantize price based on side for better fills.
    BUY orders: floor to price increment (more aggressive)
    SELL orders: ceil to price increment (more aggressive)
    Invalid sides default to BUY behavior (floor).
    """
    if increment <= 0:
        return price

    normalized = price / increment
    if side.upper() == "SELL":
        steps = normalized.to_integral_value(rounding=ROUND_UP)
    else:
        steps = normalized.to_integral_value(rounding=ROUND_DOWN)
    return (steps * increment).quantize(increment)
