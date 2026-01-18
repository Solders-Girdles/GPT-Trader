"""Property-based tests for compute_margin_state utilization and leverage invariants."""

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
    """Property: Margin utilization should never be negative."""
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

    assert snapshot.margin_utilization >= Decimal(
        "0"
    ), f"Margin utilization {snapshot.margin_utilization} should not be negative"
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
    """Property: Leverage = positions_notional / total_equity."""
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

    expected_leverage = snapshot.positions_notional / total_equity
    tolerance = Decimal("0.0001")

    assert (
        abs(snapshot.leverage - expected_leverage) < tolerance
    ), f"Leverage {snapshot.leverage} doesn't match expected {expected_leverage}"
