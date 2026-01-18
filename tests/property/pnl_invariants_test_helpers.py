"""Test helpers for property-based PnL invariants."""

from __future__ import annotations

from hypothesis import strategies as st

# Strategies for generating valid PnL parameters
pnl_value_strategy = st.floats(
    min_value=-1000000, max_value=1000000, allow_nan=False, allow_infinity=False
)

positive_pnl_strategy = st.floats(
    min_value=0.01, max_value=1000000, allow_nan=False, allow_infinity=False
)

equity_strategy = st.floats(
    min_value=100, max_value=10000000, allow_nan=False, allow_infinity=False
)


def make_pnl_event(realized: float, unrealized: float) -> dict:
    """Create a PnL update event."""
    return {
        "type": "pnl_update",
        "realized_pnl": realized,
        "unrealized_pnl": unrealized,
    }


def make_fill_event(pnl: float, fee: float = 0.0) -> dict:
    """Create a fill event."""
    return {
        "type": "fill",
        "pnl": pnl,
        "fee": fee,
        "timestamp": "2025-01-01T12:00:00",
    }


def make_funding_event(amount: float) -> dict:
    """Create a funding payment event."""
    return {
        "type": "funding_payment",
        "amount": amount,
    }
