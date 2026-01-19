"""Shared helpers for fee invariants property tests."""

from __future__ import annotations

from decimal import Decimal

from hypothesis import strategies as st

from gpt_trader.backtesting.types import FeeTier

notional_strategy = st.decimals(
    min_value="0",
    max_value="100000000",
    allow_nan=False,
    allow_infinity=False,
    places=2,
)

positive_notional_strategy = st.decimals(
    min_value="0.01",
    max_value="100000000",
    allow_nan=False,
    allow_infinity=False,
    places=2,
)

tier_strategy = st.sampled_from(list(FeeTier))

TIERS_ORDERED = [
    FeeTier.TIER_0,
    FeeTier.TIER_1,
    FeeTier.TIER_2,
    FeeTier.TIER_3,
    FeeTier.TIER_4,
    FeeTier.TIER_5,
    FeeTier.TIER_6,
    FeeTier.TIER_7,
]


def expected_tier_for_volume(volume: Decimal) -> FeeTier:
    """Determine expected tier based on cumulative volume."""
    if volume < Decimal("10000"):
        return FeeTier.TIER_0
    elif volume < Decimal("50000"):
        return FeeTier.TIER_1
    elif volume < Decimal("100000"):
        return FeeTier.TIER_2
    elif volume < Decimal("1000000"):
        return FeeTier.TIER_3
    elif volume < Decimal("15000000"):
        return FeeTier.TIER_4
    elif volume < Decimal("75000000"):
        return FeeTier.TIER_5
    elif volume < Decimal("250000000"):
        return FeeTier.TIER_6
    else:
        return FeeTier.TIER_7
