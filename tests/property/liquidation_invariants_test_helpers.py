"""Hypothesis strategies for liquidation invariant tests."""

from __future__ import annotations

from hypothesis import strategies as st

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
