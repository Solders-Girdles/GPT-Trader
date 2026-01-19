"""Property-based tests for liquidation price invariants."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import assume, given, seed, settings

from gpt_trader.monitoring.domain.perps.liquidation import LiquidationMonitor, MarginInfo
from tests.property.liquidation_invariants_test_helpers import (
    leverage_strategy,
    maintenance_margin_strategy,
    price_strategy,
    side_strategy,
)


@seed(1001)
@settings(max_examples=200, deadline=None)
@given(
    entry_price=price_strategy,
    leverage=leverage_strategy,
    maintenance_margin_rate=maintenance_margin_strategy,
    side=side_strategy,
)
def test_liquidation_price_invariants(
    entry_price: Decimal,
    leverage: Decimal,
    maintenance_margin_rate: Decimal,
    side: str,
) -> None:
    """
    Property: Liquidation price is always positioned correctly relative to entry.

    For long positions: liquidation_price < entry_price (liquidated when price drops)
    For short positions: liquidation_price > entry_price (liquidated when price rises)

    Exception: Very high leverage with high maintenance margin can violate this,
    but the formulas should still be mathematically consistent.
    """
    # Skip edge cases where math becomes degenerate
    assume(leverage > Decimal("1"))
    assume(entry_price > Decimal("0"))

    monitor = LiquidationMonitor()

    margin_info = MarginInfo(
        symbol="TEST-PERP",
        position_size=Decimal("1"),
        position_side=side,
        entry_price=entry_price,
        current_price=entry_price,  # Start at entry
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
    )

    liq_price = monitor.calculate_liquidation_price(margin_info)

    assert liq_price is not None, "Liquidation price should be calculable"
    assert liq_price > Decimal("0"), "Liquidation price should be positive"

    # Verify the mathematical formula:
    # Long: liq = entry * (1 - 1/leverage + mm_rate)
    # Short: liq = entry * (1 + 1/leverage - mm_rate)
    inv_leverage = Decimal("1") / leverage

    if side == "long":
        expected = entry_price * (Decimal("1") - inv_leverage + maintenance_margin_rate)
        # Long positions are liquidated when price drops
        # This should typically give liq < entry, unless mm_rate > 1/leverage
        if maintenance_margin_rate < inv_leverage:
            assert (
                liq_price < entry_price
            ), f"Long liquidation price {liq_price} should be below entry {entry_price}"
    else:
        expected = entry_price * (Decimal("1") + inv_leverage - maintenance_margin_rate)
        # Short positions are liquidated when price rises
        # This should typically give liq > entry, unless mm_rate > 1/leverage
        if maintenance_margin_rate < inv_leverage:
            assert (
                liq_price > entry_price
            ), f"Short liquidation price {liq_price} should be above entry {entry_price}"

    # Verify calculation matches expected formula
    assert abs(liq_price - expected) < Decimal(
        "0.0001"
    ), f"Liquidation price {liq_price} doesn't match expected {expected}"


@pytest.mark.property
class TestLiquidationPropertyBased:
    """Grouped property-based tests for liquidation monitoring."""

    def test_liquidation_price_formula_documented(self) -> None:
        """Verify the liquidation formula matches documentation."""
        # Long: liq_price = entry_price * (1 - 1/leverage + maintenance_margin_rate)
        # Short: liq_price = entry_price * (1 + 1/leverage - maintenance_margin_rate)

        entry = Decimal("50000")
        leverage = Decimal("10")  # 10x leverage
        mm_rate = Decimal("0.05")  # 5% maintenance margin

        monitor = LiquidationMonitor()

        long_margin = MarginInfo(
            symbol="BTC-PERP",
            position_size=Decimal("1"),
            position_side="long",
            entry_price=entry,
            current_price=entry,
            leverage=leverage,
            maintenance_margin_rate=mm_rate,
        )

        short_margin = MarginInfo(
            symbol="BTC-PERP",
            position_size=Decimal("1"),
            position_side="short",
            entry_price=entry,
            current_price=entry,
            leverage=leverage,
            maintenance_margin_rate=mm_rate,
        )

        # Long: 50000 * (1 - 0.1 + 0.05) = 50000 * 0.95 = 47500
        expected_long = entry * (Decimal("1") - Decimal("0.1") + Decimal("0.05"))
        actual_long = monitor.calculate_liquidation_price(long_margin)
        assert actual_long == expected_long

        # Short: 50000 * (1 + 0.1 - 0.05) = 50000 * 1.05 = 52500
        expected_short = entry * (Decimal("1") + Decimal("0.1") - Decimal("0.05"))
        actual_short = monitor.calculate_liquidation_price(short_margin)
        assert actual_short == expected_short
