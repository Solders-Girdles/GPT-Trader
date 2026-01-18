"""Property-based tests for compute_margin_state requirements and flags invariants."""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import assume, given, seed, settings

from gpt_trader.monitoring.domain.perps.margin import MarginStateMonitor
from tests.property.margin_invariants_test_helpers import (
    equity_strategy,
    price_strategy,
    quantity_strategy,
)


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
    """Property: Margin requirements should increase with more notional."""
    assume(total_equity > Decimal("0"))
    assume(price1 > Decimal("0") and price2 > Decimal("0"))
    assume(quantity1 > Decimal("0") or quantity2 > Decimal("0"))

    monitor = MarginStateMonitor()

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

    notional1 = abs(quantity1 * price1)
    notional2 = abs(quantity2 * price2)
    expected_double_notional = notional1 + notional2

    tolerance = Decimal("0.01")
    assert (
        abs(snapshot_double.positions_notional - expected_double_notional) < tolerance
    ), f"Combined notional {snapshot_double.positions_notional} doesn't match expected {expected_double_notional}"

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
    """Property: Zero positions result in zero margin usage/requirements."""
    assume(total_equity > Decimal("0"))

    monitor = MarginStateMonitor()

    snapshot = await monitor.compute_margin_state(
        total_equity=total_equity,
        cash_balance=total_equity,
        positions={},
    )

    assert snapshot.positions_notional == Decimal("0")
    assert snapshot.initial_margin_req == Decimal("0")
    assert snapshot.maintenance_margin_req == Decimal("0")
    assert snapshot.leverage == Decimal("0")
    assert snapshot.is_margin_call is False
    assert snapshot.is_liquidation_risk is False


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
    """Property: margin_available = max(0, total_equity - margin_used)."""
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

    assert snapshot.margin_available >= Decimal(
        "0"
    ), f"Margin available {snapshot.margin_available} should not be negative"

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
    """Property: margin call when equity <= maintenance requirement."""
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

    expected_margin_call = total_equity <= snapshot.maintenance_margin_req
    assert snapshot.is_margin_call == expected_margin_call, (
        f"Margin call state {snapshot.is_margin_call} inconsistent with "
        f"equity {total_equity} vs maintenance {snapshot.maintenance_margin_req}"
    )
