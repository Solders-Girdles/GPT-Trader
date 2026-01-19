"""Tests for ATR-related indicators."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.live_trade.indicators import average_true_range, true_range


class TestTrueRange:
    """Tests for true_range function."""

    def test_simple_range(self) -> None:
        tr = true_range(
            high=Decimal("105"),
            low=Decimal("100"),
            prev_close=Decimal("102"),
        )
        # Max of: 105-100=5, |105-102|=3, |100-102|=2
        assert tr == Decimal("5")

    def test_gap_up(self) -> None:
        tr = true_range(
            high=Decimal("115"),
            low=Decimal("110"),
            prev_close=Decimal("100"),
        )
        # Max of: 115-110=5, |115-100|=15, |110-100|=10
        assert tr == Decimal("15")


class TestAverageTrueRange:
    """Tests for average_true_range function."""

    def test_insufficient_data_returns_none(self) -> None:
        highs = [Decimal("100"), Decimal("101")]
        lows = [Decimal("99"), Decimal("100")]
        closes = [Decimal("100"), Decimal("101")]
        assert average_true_range(highs, lows, closes, period=14) is None

    def test_atr_with_sufficient_data(self) -> None:
        # Create synthetic OHLC data
        highs = [Decimal(str(100 + i + 2)) for i in range(20)]
        lows = [Decimal(str(100 + i - 2)) for i in range(20)]
        closes = [Decimal(str(100 + i)) for i in range(20)]
        atr = average_true_range(highs, lows, closes, period=14)
        assert atr is not None
        assert atr > Decimal("0")
