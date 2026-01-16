"""Property-based tests for liquidation calculation invariants.

Tests critical trading safety properties:
- Liquidation prices are always on the correct side of entry price
- Distance calculations are consistent with price relationships
- Risk levels are correctly determined based on distance thresholds
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import HealthCheck, assume, given, seed, settings
from hypothesis import strategies as st

from gpt_trader.monitoring.domain.perps.liquidation import (
    LiquidationMonitor,
    MarginInfo,
)

# Strategies for generating valid trading parameters
leverage_strategy = st.decimals(
    min_value="1.01", max_value="100", allow_nan=False, allow_infinity=False, places=2
)

price_strategy = st.decimals(
    min_value="0.01", max_value="1000000", allow_nan=False, allow_infinity=False, places=4
)

position_size_strategy = st.decimals(
    min_value="0.0001", max_value="10000", allow_nan=False, allow_infinity=False, places=6
)

maintenance_margin_strategy = st.decimals(
    min_value="0.001", max_value="0.5", allow_nan=False, allow_infinity=False, places=4
)

side_strategy = st.sampled_from(["long", "short"])


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


@seed(1002)
@settings(max_examples=200, deadline=None)
@given(
    current_price=price_strategy,
    liquidation_price=price_strategy,
    side=side_strategy,
)
def test_distance_calculation_invariants(
    current_price: Decimal,
    liquidation_price: Decimal,
    side: str,
) -> None:
    """
    Property: Distance to liquidation calculations are mathematically consistent.

    - distance_bps = distance_pct * 100 (basis points = percentage * 100)
    - Long: positive distance when current > liq (safe), negative when current < liq
    - Short: positive distance when liq > current (safe), negative when liq < current
    """
    assume(current_price > Decimal("0"))
    assume(liquidation_price > Decimal("0"))

    monitor = LiquidationMonitor()

    distance_pct, distance_bps = monitor.calculate_distance_to_liquidation(
        current_price, liquidation_price, side
    )

    # Invariant: bps = pct * 100
    assert (
        abs(distance_bps - (distance_pct * 100)) < 0.01
    ), f"BPS {distance_bps} should equal PCT {distance_pct} * 100"

    # Verify direction based on position side
    if side == "long":
        # Long: distance = (current - liq) / current
        expected_pct = float((current_price - liquidation_price) / current_price * 100)
        if current_price > liquidation_price:
            assert distance_pct > 0, "Long with current > liq should have positive distance"
        elif current_price < liquidation_price:
            assert distance_pct < 0, "Long with current < liq should have negative distance"
    else:
        # Short: distance = (liq - current) / current
        expected_pct = float((liquidation_price - current_price) / current_price * 100)
        if liquidation_price > current_price:
            assert distance_pct > 0, "Short with liq > current should have positive distance"
        elif liquidation_price < current_price:
            assert distance_pct < 0, "Short with liq < current should have negative distance"

    # Verify calculation matches expected
    assert (
        abs(distance_pct - expected_pct) < 0.01
    ), f"Distance PCT {distance_pct} doesn't match expected {expected_pct}"


@seed(1003)
@settings(max_examples=150, deadline=None)
@given(
    entry_price=price_strategy,
    current_price=price_strategy,
    leverage=leverage_strategy,
    position_size=position_size_strategy,
    maintenance_margin_rate=maintenance_margin_strategy,
    side=side_strategy,
    warning_buffer=st.floats(min_value=10.0, max_value=50.0),
    critical_buffer=st.floats(min_value=5.0, max_value=25.0),
)
def test_risk_level_invariants(
    entry_price: Decimal,
    current_price: Decimal,
    leverage: Decimal,
    position_size: Decimal,
    maintenance_margin_rate: Decimal,
    side: str,
    warning_buffer: float,
    critical_buffer: float,
) -> None:
    """
    Property: Risk levels follow a strict ordering based on distance.

    - safe: distance > warning_buffer
    - warning: critical_buffer < distance <= warning_buffer
    - critical: 0 < distance <= critical_buffer
    - liquidated: distance <= 0

    The guards (reduce_only, reject_entry) should be consistent with risk levels.
    """
    assume(leverage > Decimal("1"))
    assume(entry_price > Decimal("0"))
    assume(current_price > Decimal("0"))
    assume(position_size > Decimal("0"))
    # Ensure warning > critical for valid hierarchy
    assume(warning_buffer > critical_buffer)

    monitor = LiquidationMonitor(
        warning_buffer_pct=warning_buffer,
        critical_buffer_pct=critical_buffer,
        enable_reduce_only_guard=True,
        enable_entry_rejection=True,
    )

    margin_info = MarginInfo(
        symbol="TEST-PERP",
        position_size=position_size,
        position_side=side,
        entry_price=entry_price,
        current_price=current_price,
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
    )

    risk = monitor.assess_liquidation_risk(margin_info)

    # Verify risk level consistency
    if risk.distance_pct is not None:
        distance = risk.distance_pct

        if distance <= 0:
            assert (
                risk.risk_level == "liquidated"
            ), f"Distance {distance} <= 0 should be 'liquidated', got '{risk.risk_level}'"
            assert risk.should_reduce_only is True
            assert risk.should_reject_entry is True
        elif distance <= critical_buffer:
            assert (
                risk.risk_level == "critical"
            ), f"Distance {distance} <= {critical_buffer} should be 'critical', got '{risk.risk_level}'"
            assert risk.should_reduce_only is True
            assert risk.should_reject_entry is True
        elif distance <= warning_buffer:
            assert (
                risk.risk_level == "warning"
            ), f"Distance {distance} <= {warning_buffer} should be 'warning', got '{risk.risk_level}'"
            assert risk.should_reduce_only is False
            assert risk.should_reject_entry is True
        else:
            assert (
                risk.risk_level == "safe"
            ), f"Distance {distance} > {warning_buffer} should be 'safe', got '{risk.risk_level}'"
            assert risk.should_reduce_only is False
            assert risk.should_reject_entry is False


@seed(1004)
@settings(max_examples=100, deadline=None)
@given(
    entry_price=price_strategy,
    leverage=leverage_strategy,
    maintenance_margin_rate=maintenance_margin_strategy,
    side=side_strategy,
)
def test_zero_position_is_always_safe(
    entry_price: Decimal,
    leverage: Decimal,
    maintenance_margin_rate: Decimal,
    side: str,
) -> None:
    """
    Property: Zero position size should always result in 'safe' risk level.

    This is a critical safety property - we should never flag risk on empty positions.
    """
    monitor = LiquidationMonitor()

    margin_info = MarginInfo(
        symbol="TEST-PERP",
        position_size=Decimal("0"),  # Zero position
        position_side=side,
        entry_price=entry_price,
        current_price=entry_price,
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
    )

    risk = monitor.assess_liquidation_risk(margin_info)

    assert risk.risk_level == "safe", f"Zero position should be safe, got '{risk.risk_level}'"
    assert risk.should_reduce_only is False
    assert risk.should_reject_entry is False


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
