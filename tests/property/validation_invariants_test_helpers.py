"""Shared strategies for validation invariants property tests."""

from __future__ import annotations

from hypothesis import strategies as st

# Strategies for generating valid parameters
check_type_strategy = st.sampled_from(
    [
        "mark_staleness",
        "slippage_guard",
        "order_preview",
        "api_health",
        "broker_read",
    ]
)
threshold_strategy = st.integers(min_value=1, max_value=20)
operation_count_strategy = st.integers(min_value=1, max_value=50)
