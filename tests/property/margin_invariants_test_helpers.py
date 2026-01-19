"""Test helpers for property-based margin invariants."""

from __future__ import annotations

from hypothesis import strategies as st

equity_strategy = st.decimals(
    min_value="100", max_value="10000000", allow_nan=False, allow_infinity=False, places=2
)

price_strategy = st.decimals(
    min_value="0.01", max_value="1000000", allow_nan=False, allow_infinity=False, places=4
)

quantity_strategy = st.decimals(
    min_value="0", max_value="10000", allow_nan=False, allow_infinity=False, places=6
)

hour_strategy = st.integers(min_value=0, max_value=23)
minute_strategy = st.integers(min_value=0, max_value=59)
