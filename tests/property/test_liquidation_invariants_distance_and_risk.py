"""Property-based tests for liquidation distance calculations and risk levels."""

from __future__ import annotations

from decimal import Decimal

from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from gpt_trader.monitoring.domain.perps.liquidation import LiquidationMonitor, MarginInfo
from tests.property.liquidation_invariants_test_helpers import (
    leverage_strategy,
    maintenance_margin_strategy,
    position_size_strategy,
    price_strategy,
    side_strategy,
)


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
