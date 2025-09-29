from decimal import Decimal

import pytest

from bot_v2.utilities.quantization import (
    quantize_price,
    quantize_price_side_aware,
    quantize_size,
)


@pytest.mark.parametrize(
    "size,step,min_size,expected",
    [
        (Decimal("1.2345"), Decimal("0.001"), Decimal("0.01"), Decimal("1.235")),
        (Decimal("0.0001"), Decimal("0.001"), Decimal("0.01"), Decimal("0.01")),
        (Decimal("-0.5"), Decimal("0.001"), Decimal("0.01"), Decimal("0")),
        (Decimal("0.2"), Decimal("0"), Decimal("0.01"), Decimal("0.2")),
    ],
)
def test_quantize_size(size: Decimal, step: Decimal, min_size: Decimal, expected: Decimal) -> None:
    assert quantize_size(size, step, min_size) == expected


@pytest.mark.parametrize(
    "price,increment,expected",
    [
        (Decimal("123.456"), Decimal("0.01"), Decimal("123.46")),
        (Decimal("50"), Decimal("0.5"), Decimal("50.0")),
        (Decimal("10"), Decimal("0"), Decimal("10")),
    ],
)
def test_quantize_price(price: Decimal, increment: Decimal, expected: Decimal) -> None:
    assert quantize_price(price, increment) == expected


@pytest.mark.parametrize(
    "price,increment,side,expected",
    [
        (Decimal("123.456"), Decimal("0.01"), "buy", Decimal("123.45")),
        (Decimal("123.456"), Decimal("0.01"), "sell", Decimal("123.46")),
        (Decimal("123.456"), Decimal("0.01"), "SeLl", Decimal("123.46")),
        (Decimal("123"), Decimal("0"), "buy", Decimal("123")),
    ],
)
def test_quantize_price_side_aware(
    price: Decimal, increment: Decimal, side: str, expected: Decimal
) -> None:
    assert quantize_price_side_aware(price, increment, side) == expected
