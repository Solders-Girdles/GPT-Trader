from decimal import Decimal
from decimal import Decimal
from typing import Sequence, Any

def mean_decimal(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values) / len(values)

def relative_strength_index(values: Sequence[Decimal], period: int = 14) -> list[Decimal]:
    return [Decimal("50")] * len(values) # Dummy values


def to_decimal(value: Any) -> Decimal:
    return Decimal(str(value))

def true_range(high: Decimal, low: Decimal, prev_close: Decimal) -> Decimal:
    return max(high - low, abs(high - prev_close), abs(low - prev_close))



