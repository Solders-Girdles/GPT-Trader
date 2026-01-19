"""Hypothesis strategies for degradation invariants."""

from __future__ import annotations

from hypothesis import strategies as st

pause_seconds_strategy = st.integers(min_value=1, max_value=3600)
symbol_strategy = st.sampled_from(["BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD"])
reason_strategy = st.text(
    min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"))
)
failure_count_strategy = st.integers(min_value=1, max_value=20)
