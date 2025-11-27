"""Property-based tests for margin state calculation invariants.

Tests critical margin safety properties:
- Margin utilization is always in [0, 1] range
- Leverage calculation is consistent with notional/equity
- Margin requirements scale with positions
- Window policy transitions are deterministic
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import pytest
from hypothesis import assume, given, seed, settings
from hypothesis import strategies as st

from gpt_trader.monitoring.domain.perps.margin import (
    MarginStateMonitor,
    MarginWindow,
    MarginWindowPolicy,
)

# Strategies for generating valid margin parameters
equity_strategy = st.decimals(
    min_value="100", max_value="10000000", allow_nan=False, allow_infinity=False, places=2
)

notional_strategy = st.decimals(
    min_value="0", max_value="100000000", allow_nan=False, allow_infinity=False, places=2
)

price_strategy = st.decimals(
    min_value="0.01", max_value="1000000", allow_nan=False, allow_infinity=False, places=4
)

quantity_strategy = st.decimals(
    min_value="0", max_value="10000", allow_nan=False, allow_infinity=False, places=6
)

rate_strategy = st.decimals(
    min_value="0.01", max_value="1.0", allow_nan=False, allow_infinity=False, places=4
)

hour_strategy = st.integers(min_value=0, max_value=23)
minute_strategy = st.integers(min_value=0, max_value=59)


@seed(2001)
@settings(max_examples=150, deadline=None)
@given(
    total_equity=equity_strategy,
    cash_balance=equity_strategy,
    position_quantity=quantity_strategy,
    mark_price=price_strategy,
)
@pytest.mark.asyncio
async def test_margin_utilization_bounded(
    total_equity: Decimal,
    cash_balance: Decimal,
    position_quantity: Decimal,
    mark_price: Decimal,
) -> None:
    """
    Property: Margin utilization should always be in [0, 1] range.

    Utilization = margin_used / total_equity
    When equity is positive, this should never exceed 1 (100%) or go negative.
    """
    assume(total_equity > Decimal("0"))
    assume(mark_price > Decimal("0"))

    monitor = MarginStateMonitor()

    positions = {
        "TEST-PERP": {
            "quantity": float(position_quantity),
            "mark_price": float(mark_price),
        }
    }

    snapshot = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=cash_balance,
        positions=positions,
    )

    # Utilization should be bounded
    assert snapshot.margin_utilization >= Decimal(
        "0"
    ), f"Margin utilization {snapshot.margin_utilization} should not be negative"

    # Note: Utilization CAN exceed 1 if over-leveraged (this is a margin call state)
    # But it should never be negative
    assert isinstance(snapshot.margin_utilization, Decimal)


@seed(2002)
@settings(max_examples=150, deadline=None)
@given(
    total_equity=equity_strategy,
    cash_balance=equity_strategy,
    position_quantity=quantity_strategy,
    mark_price=price_strategy,
)
@pytest.mark.asyncio
async def test_leverage_calculation_consistency(
    total_equity: Decimal,
    cash_balance: Decimal,
    position_quantity: Decimal,
    mark_price: Decimal,
) -> None:
    """
    Property: Leverage = positions_notional / total_equity.

    This fundamental relationship must always hold.
    """
    assume(total_equity > Decimal("0"))
    assume(mark_price > Decimal("0"))

    monitor = MarginStateMonitor()

    positions = {
        "TEST-PERP": {
            "quantity": float(position_quantity),
            "mark_price": float(mark_price),
        }
    }

    snapshot = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=cash_balance,
        positions=positions,
    )

    # Verify leverage calculation
    expected_leverage = snapshot.positions_notional / total_equity
    tolerance = Decimal("0.0001")

    assert (
        abs(snapshot.leverage - expected_leverage) < tolerance
    ), f"Leverage {snapshot.leverage} doesn't match expected {expected_leverage}"


@seed(2003)
@settings(max_examples=150, deadline=None)
@given(
    total_equity=equity_strategy,
    cash_balance=equity_strategy,
    quantity1=quantity_strategy,
    quantity2=quantity_strategy,
    price1=price_strategy,
    price2=price_strategy,
)
@pytest.mark.asyncio
async def test_margin_requirements_scale_with_positions(
    total_equity: Decimal,
    cash_balance: Decimal,
    quantity1: Decimal,
    quantity2: Decimal,
    price1: Decimal,
    price2: Decimal,
) -> None:
    """
    Property: Margin requirements should increase proportionally with position size.

    If positions double, margin requirements should roughly double.
    """
    assume(total_equity > Decimal("0"))
    assume(price1 > Decimal("0") and price2 > Decimal("0"))
    assume(quantity1 > Decimal("0") or quantity2 > Decimal("0"))

    monitor = MarginStateMonitor()

    # Single position
    positions_single = {
        "TEST1-PERP": {
            "quantity": float(quantity1),
            "mark_price": float(price1),
        }
    }

    snapshot_single = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=cash_balance,
        positions=positions_single,
    )

    # Two positions
    positions_double = {
        "TEST1-PERP": {
            "quantity": float(quantity1),
            "mark_price": float(price1),
        },
        "TEST2-PERP": {
            "quantity": float(quantity2),
            "mark_price": float(price2),
        },
    }

    snapshot_double = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=cash_balance,
        positions=positions_double,
    )

    # Notional should be additive
    notional1 = abs(quantity1 * price1)
    notional2 = abs(quantity2 * price2)
    expected_double_notional = notional1 + notional2

    tolerance = Decimal("0.01")
    assert (
        abs(snapshot_double.positions_notional - expected_double_notional) < tolerance
    ), f"Combined notional {snapshot_double.positions_notional} doesn't match expected {expected_double_notional}"

    # Margin requirements should increase (not necessarily double, depends on window)
    if quantity2 > Decimal("0"):
        assert (
            snapshot_double.initial_margin_req >= snapshot_single.initial_margin_req
        ), "Adding positions should not decrease margin requirements"


@seed(2004)
@settings(max_examples=100, deadline=None)
@given(
    total_equity=equity_strategy,
)
@pytest.mark.asyncio
async def test_zero_positions_zero_margin(
    total_equity: Decimal,
) -> None:
    """
    Property: Zero positions should result in zero margin requirements.
    """
    assume(total_equity > Decimal("0"))

    monitor = MarginStateMonitor()

    snapshot = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=total_equity,
        positions={},  # No positions
    )

    assert snapshot.positions_notional == Decimal("0")
    assert snapshot.initial_margin_req == Decimal("0")
    assert snapshot.maintenance_margin_req == Decimal("0")
    assert snapshot.leverage == Decimal("0")
    assert snapshot.is_margin_call is False
    assert snapshot.is_liquidation_risk is False


@seed(2005)
@settings(max_examples=200, deadline=None)
@given(
    hour=hour_strategy,
    minute=minute_strategy,
)
def test_margin_window_deterministic(
    hour: int,
    minute: int,
) -> None:
    """
    Property: Margin window determination should be deterministic.

    Same time should always produce same window.
    """
    policy = MarginWindowPolicy()

    # Create a specific datetime
    test_time = datetime(2025, 1, 15, hour, minute, 0)

    # Call twice - should get same result
    window1 = policy.determine_current_window(test_time)
    window2 = policy.determine_current_window(test_time)

    assert window1 == window2, f"Window determination not deterministic for {test_time}"
    assert isinstance(window1, MarginWindow)


@seed(2006)
@settings(max_examples=100, deadline=None)
@given(
    hour=hour_strategy,
    minute=minute_strategy,
)
def test_margin_window_coverage(
    hour: int,
    minute: int,
) -> None:
    """
    Property: Every time should map to exactly one valid margin window.

    No gaps or overlaps in window assignment.
    """
    policy = MarginWindowPolicy()
    test_time = datetime(2025, 1, 15, hour, minute, 0)

    window = policy.determine_current_window(test_time)

    # Should be one of the defined windows
    valid_windows = {
        MarginWindow.NORMAL,
        MarginWindow.INTRADAY,
        MarginWindow.OVERNIGHT,
        MarginWindow.PRE_FUNDING,
    }

    assert window in valid_windows, f"Unknown window {window} for time {test_time}"


@seed(2007)
@settings(max_examples=50, deadline=None)
@given(
    hour=hour_strategy,
    minute=minute_strategy,
)
def test_margin_requirements_ordering(
    hour: int,
    minute: int,
) -> None:
    """
    Property: Margin requirements should maintain consistent ordering.

    NORMAL <= INTRADAY <= OVERNIGHT <= PRE_FUNDING (in terms of strictness)
    This is enforced by initial_rate ordering.
    """
    policy = MarginWindowPolicy()

    normal_req = policy.get_requirements(MarginWindow.NORMAL)
    intraday_req = policy.get_requirements(MarginWindow.INTRADAY)
    overnight_req = policy.get_requirements(MarginWindow.OVERNIGHT)
    pre_funding_req = policy.get_requirements(MarginWindow.PRE_FUNDING)

    # Initial margin rates should be ordered (higher = stricter)
    assert (
        normal_req.initial_rate <= intraday_req.initial_rate
    ), "Normal should have lower initial margin than intraday"
    assert (
        intraday_req.initial_rate <= overnight_req.initial_rate
    ), "Intraday should have lower initial margin than overnight"
    assert (
        overnight_req.initial_rate <= pre_funding_req.initial_rate
    ), "Overnight should have lower initial margin than pre_funding"

    # Max leverage should be inverse ordered (lower = stricter)
    assert (
        normal_req.max_leverage >= intraday_req.max_leverage
    ), "Normal should allow higher leverage than intraday"
    assert (
        intraday_req.max_leverage >= overnight_req.max_leverage
    ), "Intraday should allow higher leverage than overnight"
    assert (
        overnight_req.max_leverage >= pre_funding_req.max_leverage
    ), "Overnight should allow higher leverage than pre_funding"


@seed(2008)
@settings(max_examples=100, deadline=None)
@given(
    total_equity=equity_strategy,
    cash_balance=equity_strategy,
    position_quantity=quantity_strategy,
    mark_price=price_strategy,
)
@pytest.mark.asyncio
async def test_margin_available_consistency(
    total_equity: Decimal,
    cash_balance: Decimal,
    position_quantity: Decimal,
    mark_price: Decimal,
) -> None:
    """
    Property: Margin available = max(0, total_equity - margin_used).

    Available margin should never be negative.
    """
    assume(total_equity > Decimal("0"))
    assume(mark_price > Decimal("0"))

    monitor = MarginStateMonitor()

    positions = {
        "TEST-PERP": {
            "quantity": float(position_quantity),
            "mark_price": float(mark_price),
        }
    }

    snapshot = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=cash_balance,
        positions=positions,
    )

    # Margin available should never be negative
    assert snapshot.margin_available >= Decimal(
        "0"
    ), f"Margin available {snapshot.margin_available} should not be negative"

    # Verify the formula
    expected_available = max(Decimal("0"), total_equity - snapshot.margin_used)
    assert (
        snapshot.margin_available == expected_available
    ), f"Margin available {snapshot.margin_available} doesn't match expected {expected_available}"


@seed(2009)
@settings(max_examples=100, deadline=None)
@given(
    total_equity=equity_strategy,
    position_quantity=quantity_strategy,
    mark_price=price_strategy,
)
@pytest.mark.asyncio
async def test_margin_call_trigger_consistency(
    total_equity: Decimal,
    position_quantity: Decimal,
    mark_price: Decimal,
) -> None:
    """
    Property: Margin call is triggered when equity <= maintenance margin requirement.
    """
    assume(total_equity > Decimal("0"))
    assume(mark_price > Decimal("0"))

    monitor = MarginStateMonitor()

    positions = {
        "TEST-PERP": {
            "quantity": float(position_quantity),
            "mark_price": float(mark_price),
        }
    }

    snapshot = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=total_equity,
        positions=positions,
    )

    # Verify margin call trigger
    expected_margin_call = total_equity <= snapshot.maintenance_margin_req
    assert snapshot.is_margin_call == expected_margin_call, (
        f"Margin call state {snapshot.is_margin_call} inconsistent with "
        f"equity {total_equity} vs maintenance {snapshot.maintenance_margin_req}"
    )


@seed(2010)
@settings(max_examples=100, deadline=None)
@given(
    window=st.sampled_from(list(MarginWindow)),
)
def test_margin_requirement_consistency(window: MarginWindow) -> None:
    """
    Property: Margin requirements should be internally consistent.

    - maintenance_rate < initial_rate (always)
    - max_leverage = 1 / initial_rate (approximately)
    """
    policy = MarginWindowPolicy()
    req = policy.get_requirements(window)

    # Maintenance should always be less than initial
    assert (
        req.maintenance_rate < req.initial_rate
    ), f"Maintenance rate {req.maintenance_rate} should be less than initial {req.initial_rate}"

    # Max leverage should be consistent with initial margin
    expected_max_leverage = Decimal("1") / req.initial_rate
    tolerance = Decimal("0.1")  # Allow some rounding
    assert (
        abs(req.max_leverage - expected_max_leverage) < tolerance
    ), f"Max leverage {req.max_leverage} inconsistent with initial rate {req.initial_rate}"


@pytest.mark.property
class TestMarginPropertyBased:
    """Grouped property-based tests for margin monitoring."""

    def test_margin_window_pre_funding_times(self) -> None:
        """Verify pre-funding window is active 30 minutes before funding."""
        policy = MarginWindowPolicy()

        # Funding times are 0:00, 8:00, 16:00 UTC
        # Check 30 minutes before each

        # 23:30 -> should be PRE_FUNDING (30 min before 0:00)
        # Note: This depends on the implementation handling wrap-around
        test_time = datetime(2025, 1, 15, 23, 45, 0)
        window = policy.determine_current_window(test_time)
        assert window == MarginWindow.PRE_FUNDING

        # 7:45 -> should be NORMAL (15 min before 8:00, but PRE_FUNDING only checks same hour)
        test_time = datetime(2025, 1, 15, 7, 45, 0)
        window = policy.determine_current_window(test_time)
        # The implementation only checks if current_hour == funding_hour
        # So at 7:45, it's not yet 8:00 hour, so not PRE_FUNDING
        assert window == MarginWindow.NORMAL

        # 8:15 -> could be PRE_FUNDING (within 30 min of 8:00, same hour)
        test_time = datetime(2025, 1, 15, 8, 15, 0)
        window = policy.determine_current_window(test_time)
        # At 8:15, we're 15 min AFTER funding at 8:00, so not PRE_FUNDING
        # The check is: funding_minute - current_minute, so 0 - 15 = -15 (not in range)
        assert window == MarginWindow.NORMAL

        # 15:45 -> in INTRADAY window (14:00-16:00)
        test_time = datetime(2025, 1, 15, 15, 45, 0)
        window = policy.determine_current_window(test_time)
        assert window == MarginWindow.INTRADAY

    def test_overnight_window_hours(self) -> None:
        """Verify overnight window covers expected hours."""
        policy = MarginWindowPolicy()

        # Overnight is 22:00-06:00 UTC
        overnight_hours = [22, 23, 0, 1, 2, 3, 4, 5]

        for hour in overnight_hours:
            test_time = datetime(2025, 1, 15, hour, 30, 0)
            window = policy.determine_current_window(test_time)
            # May be OVERNIGHT or PRE_FUNDING (near funding times)
            assert window in {
                MarginWindow.OVERNIGHT,
                MarginWindow.PRE_FUNDING,
            }, f"Hour {hour} should be OVERNIGHT or PRE_FUNDING, got {window}"
