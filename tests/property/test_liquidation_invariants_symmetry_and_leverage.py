"""Property-based tests for liquidation symmetry and leverage relationships."""

from __future__ import annotations

from decimal import Decimal

from hypothesis import HealthCheck, assume, given, seed, settings
from hypothesis import strategies as st

from gpt_trader.monitoring.domain.perps.liquidation import LiquidationMonitor, MarginInfo
from tests.property.liquidation_invariants_test_helpers import (
    leverage_strategy,
    maintenance_margin_strategy,
    price_strategy,
    side_strategy,
)


@seed(1005)
@settings(max_examples=100, deadline=None)
@given(
    entry_price=price_strategy,
    leverage=leverage_strategy,
    maintenance_margin_rate=maintenance_margin_strategy,
)
def test_liquidation_symmetry(
    entry_price: Decimal,
    leverage: Decimal,
    maintenance_margin_rate: Decimal,
) -> None:
    """
    Property: Long and short liquidation prices should be symmetric around entry.

    If a long is liquidated X% below entry, a short should be liquidated X% above entry
    (with same leverage and margin rate).
    """
    assume(leverage > Decimal("1"))
    assume(entry_price > Decimal("0"))
    assume(maintenance_margin_rate < Decimal("1") / leverage)  # Valid margin rate

    monitor = LiquidationMonitor()

    long_margin = MarginInfo(
        symbol="TEST-PERP",
        position_size=Decimal("1"),
        position_side="long",
        entry_price=entry_price,
        current_price=entry_price,
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
    )

    short_margin = MarginInfo(
        symbol="TEST-PERP",
        position_size=Decimal("1"),
        position_side="short",
        entry_price=entry_price,
        current_price=entry_price,
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
    )

    long_liq = monitor.calculate_liquidation_price(long_margin)
    short_liq = monitor.calculate_liquidation_price(short_margin)

    assert long_liq is not None and short_liq is not None

    # Calculate distances from entry
    long_distance = entry_price - long_liq
    short_distance = short_liq - entry_price

    # The distances should be equal (symmetric)
    tolerance = entry_price * Decimal("0.0001")  # 0.01% tolerance
    assert (
        abs(long_distance - short_distance) < tolerance
    ), f"Liquidation not symmetric: long_dist={long_distance}, short_dist={short_distance}"


@seed(1006)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.filter_too_much])
@given(
    entry_price=price_strategy,
    leverage1=st.decimals(
        min_value="2", max_value="20", allow_nan=False, allow_infinity=False, places=1
    ),
    leverage2=st.decimals(
        min_value="2", max_value="20", allow_nan=False, allow_infinity=False, places=1
    ),
    maintenance_margin_rate=st.decimals(
        min_value="0.01", max_value="0.03", allow_nan=False, allow_infinity=False, places=3
    ),
    side=side_strategy,
)
def test_higher_leverage_closer_liquidation(
    entry_price: Decimal,
    leverage1: Decimal,
    leverage2: Decimal,
    maintenance_margin_rate: Decimal,
    side: str,
) -> None:
    """
    Property: Higher leverage should result in liquidation price closer to entry.

    This is fundamental to leveraged trading - more leverage = more risk = tighter liquidation.
    """
    assume(leverage1 != leverage2)
    assume(entry_price > Decimal("0"))

    # Ensure margin rate doesn't dominate the formula (mm_rate < 1/max_leverage)
    max_leverage = max(leverage1, leverage2)
    assume(maintenance_margin_rate < Decimal("1") / max_leverage)

    monitor = LiquidationMonitor()

    margin_low_lev = MarginInfo(
        symbol="TEST-PERP",
        position_size=Decimal("1"),
        position_side=side,
        entry_price=entry_price,
        current_price=entry_price,
        leverage=min(leverage1, leverage2),
        maintenance_margin_rate=maintenance_margin_rate,
    )

    margin_high_lev = MarginInfo(
        symbol="TEST-PERP",
        position_size=Decimal("1"),
        position_side=side,
        entry_price=entry_price,
        current_price=entry_price,
        leverage=max(leverage1, leverage2),
        maintenance_margin_rate=maintenance_margin_rate,
    )

    liq_low = monitor.calculate_liquidation_price(margin_low_lev)
    liq_high = monitor.calculate_liquidation_price(margin_high_lev)

    assert liq_low is not None and liq_high is not None

    # Calculate distances from entry
    dist_low = abs(entry_price - liq_low)
    dist_high = abs(entry_price - liq_high)

    # Higher leverage should have smaller distance (closer liquidation)
    assert dist_high < dist_low, (
        f"Higher leverage ({max(leverage1, leverage2)}) should have closer liquidation. "
        f"Low lev dist={dist_low}, High lev dist={dist_high}"
    )
