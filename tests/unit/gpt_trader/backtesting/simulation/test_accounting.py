"""Unit tests for the pure simulated-broker accounting helpers."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.backtesting.simulation.accounting import (
    calculate_margin_used,
    credit_pnl,
    lock_margin,
    release_margin,
)
from gpt_trader.core import Balance, Position


def _position(quantity: str, mark: str, leverage: int | None) -> Position:
    return Position(
        symbol="BTC-USD",
        quantity=Decimal(quantity),
        entry_price=Decimal(mark),
        mark_price=Decimal(mark),
        unrealized_pnl=Decimal("0"),
        realized_pnl=Decimal("0"),
        side="long" if Decimal(quantity) >= 0 else "short",
        leverage=leverage,
    )


def test_calculate_margin_used_sums_notional_over_leverage() -> None:
    positions = {
        "A": _position("2", "100", leverage=2),  # 200 / 2 = 100
        "B": _position("-1", "50", leverage=None),  # 50 / 1 = 50
    }
    assert calculate_margin_used(positions) == Decimal("150")


def test_calculate_margin_used_empty() -> None:
    assert calculate_margin_used({}) == Decimal("0")


def test_lock_margin_moves_available_to_hold() -> None:
    balance = Balance(
        asset="USDC", total=Decimal("1000"), available=Decimal("1000"), hold=Decimal("0")
    )
    result = lock_margin(balance, Decimal("400"), leverage=2)  # margin = 200
    assert result.available == Decimal("800")
    assert result.hold == Decimal("200")
    assert result.total == Decimal("1000")  # unchanged
    assert balance.available == Decimal("1000")  # input not mutated


def test_lock_margin_defaults_leverage_to_one() -> None:
    balance = Balance(
        asset="USDC", total=Decimal("1000"), available=Decimal("1000"), hold=Decimal("0")
    )
    result = lock_margin(balance, Decimal("300"), leverage=None)
    assert result.hold == Decimal("300")


def test_release_margin_floors_hold_at_zero() -> None:
    balance = Balance(
        asset="USDC", total=Decimal("1000"), available=Decimal("800"), hold=Decimal("100")
    )
    result = release_margin(balance, Decimal("400"), leverage=2)  # margin = 200 > hold
    assert result.available == Decimal("1000")
    assert result.hold == Decimal("0")  # floored, not negative


def test_credit_pnl_adjusts_total_and_available_only() -> None:
    balance = Balance(
        asset="USDC", total=Decimal("1000"), available=Decimal("800"), hold=Decimal("200")
    )
    result = credit_pnl(balance, Decimal("-50"))
    assert result.total == Decimal("950")
    assert result.available == Decimal("750")
    assert result.hold == Decimal("200")  # unchanged
