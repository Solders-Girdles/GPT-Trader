"""Pure account and margin arithmetic for the simulated broker.

Stateless helpers that compute margin figures and new balances from explicit
inputs. ``SimulatedBroker`` owns the mutable account state and delegates the
arithmetic here, so the math is independently testable and separated from order
orchestration (mirroring the existing FeeCalculator / OrderFillModel split).
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from gpt_trader.core import Balance, Position


def calculate_margin_used(positions: Mapping[str, Position]) -> Decimal:
    """Total margin used across open positions (notional / leverage)."""
    total_margin = Decimal("0")
    for pos in positions.values():
        notional = abs(pos.quantity) * pos.mark_price
        leverage = pos.leverage or 1
        total_margin += notional / Decimal(leverage)
    return total_margin


def lock_margin(balance: Balance, notional: Decimal, leverage: int | None) -> Balance:
    """Return a new balance with position margin moved from available to hold."""
    margin = notional / Decimal(leverage or 1)
    return Balance(
        asset=balance.asset,
        total=balance.total,
        available=balance.available - margin,
        hold=balance.hold + margin,
    )


def release_margin(balance: Balance, notional: Decimal, leverage: int | None) -> Balance:
    """Return a new balance with closed-position margin released from hold."""
    margin = notional / Decimal(leverage or 1)
    return Balance(
        asset=balance.asset,
        total=balance.total,
        available=balance.available + margin,
        hold=max(Decimal("0"), balance.hold - margin),
    )


def credit_pnl(balance: Balance, pnl: Decimal) -> Balance:
    """Return a new balance with realized PnL credited to total and available."""
    return Balance(
        asset=balance.asset,
        total=balance.total + pnl,
        available=balance.available + pnl,
        hold=balance.hold,
    )
