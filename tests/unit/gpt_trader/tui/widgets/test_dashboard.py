"""Tests for TUI dashboard widgets."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.tui.widgets.dashboard import calculate_price_change_percent


class TestCalculatePriceChangePercent:
    """Tests for the calculate_price_change_percent helper function."""

    def test_positive_change(self) -> None:
        """Test calculating positive price change."""
        # Price went from 100 to 110 = +10%
        result = calculate_price_change_percent(110.0, [Decimal("100")])
        assert result == pytest.approx(10.0)

    def test_negative_change(self) -> None:
        """Test calculating negative price change."""
        # Price went from 100 to 90 = -10%
        result = calculate_price_change_percent(90.0, [Decimal("100")])
        assert result == pytest.approx(-10.0)

    def test_no_change(self) -> None:
        """Test when price hasn't changed."""
        result = calculate_price_change_percent(100.0, [Decimal("100")])
        assert result == pytest.approx(0.0)

    def test_empty_history_returns_zero(self) -> None:
        """Test that empty history returns 0.0."""
        result = calculate_price_change_percent(100.0, [])
        assert result == 0.0

    def test_zero_oldest_price_returns_zero(self) -> None:
        """Test that zero oldest price returns 0.0 to avoid division by zero."""
        result = calculate_price_change_percent(100.0, [Decimal("0")])
        assert result == 0.0

    def test_uses_oldest_price_from_history(self) -> None:
        """Test that the oldest (first) price in history is used."""
        # History: [100, 105, 110], current: 120
        # Change should be from 100 to 120 = +20%
        history = [Decimal("100"), Decimal("105"), Decimal("110")]
        result = calculate_price_change_percent(120.0, history)
        assert result == pytest.approx(20.0)

    def test_with_float_history(self) -> None:
        """Test calculation works with float history (not just Decimal)."""
        result = calculate_price_change_percent(110.0, [100.0])
        assert result == pytest.approx(10.0)

    def test_small_price_change(self) -> None:
        """Test small percentage changes are calculated correctly."""
        # Price went from 100 to 100.5 = +0.5%
        result = calculate_price_change_percent(100.5, [Decimal("100")])
        assert result == pytest.approx(0.5)

    def test_large_price_change(self) -> None:
        """Test large percentage changes are calculated correctly."""
        # Price went from 100 to 200 = +100%
        result = calculate_price_change_percent(200.0, [Decimal("100")])
        assert result == pytest.approx(100.0)

    def test_crypto_realistic_values(self) -> None:
        """Test with realistic crypto price values."""
        # BTC: $95,000 -> $97,000 = +2.1%
        result = calculate_price_change_percent(97000.0, [Decimal("95000")])
        expected = ((97000 - 95000) / 95000) * 100
        assert result == pytest.approx(expected)

    def test_low_price_asset(self) -> None:
        """Test with low-priced assets like DOGE."""
        # DOGE: $0.35 -> $0.38 = +8.57%
        result = calculate_price_change_percent(0.38, [Decimal("0.35")])
        expected = ((0.38 - 0.35) / 0.35) * 100
        assert result == pytest.approx(expected)
