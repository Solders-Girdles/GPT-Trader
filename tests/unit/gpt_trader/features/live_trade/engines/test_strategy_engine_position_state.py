"""Tests for TradingEngine position state formatting."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import Balance, Position


def test_build_position_state(engine):
    """Test position state formatting."""
    positions = {
        "BTC-USD": Position(
            symbol="BTC-USD",
            quantity=Decimal("0.5"),
            entry_price=Decimal("40000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    }

    state = engine._build_position_state("BTC-USD", positions)
    assert state["quantity"] == Decimal("0.5")
    assert state["entry_price"] == Decimal("40000")
    assert state["side"] == "long"

    state_none = engine._build_position_state("ETH-USD", positions)
    assert state_none is None


@pytest.mark.asyncio
async def test_position_state_passed_to_strategy(engine):
    """Verify strategy receives correct position state."""
    engine.context.broker.list_positions.return_value = [
        Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            entry_price=Decimal("45000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            side="long",
        )
    ]
    engine.context.broker.list_balances.return_value = [
        Balance(asset="USD", total=Decimal("10000"), available=Decimal("10000"))
    ]

    await engine._cycle()

    call_args = engine.strategy.decide.call_args
    position_state = call_args.kwargs["position_state"]

    assert position_state is not None
    assert position_state["quantity"] == Decimal("1.0")
    assert position_state["entry_price"] == Decimal("45000")
    assert position_state["side"] == "long"
