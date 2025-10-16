from __future__ import annotations

from decimal import Decimal

from bot_v2.features.live_trade.indicators import (
    mean_decimal,
    relative_strength_index,
    to_decimal,
    true_range,
)


def test_to_decimal_handles_none_and_strings():
    assert to_decimal(None) == Decimal("0")
    assert to_decimal("1.23") == Decimal("1.23")


def test_mean_decimal_rounding():
    values = [Decimal("1.111111115"), Decimal("1.111111116"), Decimal("1.111111117")]
    result = mean_decimal(values)
    assert result == Decimal("1.11111112")


def test_true_range_variants():
    high = Decimal("110")
    low = Decimal("100")
    prev_close = Decimal("105")

    tr = true_range(high, low, prev_close)
    assert tr == Decimal("10")

    tr_no_prev = true_range(high, low, None)
    assert tr_no_prev == Decimal("10")


def test_relative_strength_index_balanced_moves():
    closes = [
        Decimal("100"),
        Decimal("101"),
        Decimal("102"),
        Decimal("101"),
        Decimal("100"),
        Decimal("99"),
    ]
    rsi = relative_strength_index(closes)
    assert rsi < Decimal("50")


def test_relative_strength_index_all_gains():
    closes = [Decimal("1"), Decimal("2"), Decimal("3"), Decimal("4")]
    rsi = relative_strength_index(closes)
    assert rsi == Decimal("100")
