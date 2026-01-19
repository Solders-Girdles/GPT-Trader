"""Property-based tests for ValidationFailureTracker interleaved operations."""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker
from tests.property.validation_invariants_test_helpers import (
    check_type_strategy,
    threshold_strategy,
)


@seed(4010)
@settings(max_examples=50, deadline=None)
@given(
    operations=st.lists(
        st.tuples(
            check_type_strategy,
            st.sampled_from(["fail", "success"]),
        ),
        min_size=5,
        max_size=50,
    ),
    threshold=threshold_strategy,
)
def test_interleaved_operations_consistency(
    operations: list[tuple[str, str]],
    threshold: int,
) -> None:
    """
    Property: Interleaved fail/success operations maintain counter consistency.

    After any sequence of operations, the state should be internally consistent.
    """
    tracker = ValidationFailureTracker(escalation_threshold=threshold)

    expected_counts: dict[str, int] = defaultdict(int)

    for check_type, op in operations:
        if op == "fail":
            tracker.record_failure(check_type)
            expected_counts[check_type] += 1
        else:
            tracker.record_success(check_type)
            expected_counts[check_type] = 0

    for check_type in {ct for ct, _ in operations}:
        actual = tracker.get_failure_count(check_type)
        expected = expected_counts[check_type]
        assert actual == expected, f"Check type {check_type}: expected {expected}, got {actual}"


@seed(4011)
@settings(max_examples=50, deadline=None)
@given(
    check_type=check_type_strategy,
    threshold=threshold_strategy,
)
def test_callback_exception_does_not_break_tracking(
    check_type: str,
    threshold: int,
) -> None:
    """
    Property: If callback raises an exception, tracking should continue.
    """
    callback = MagicMock(side_effect=RuntimeError("Callback failed"))
    tracker = ValidationFailureTracker(
        escalation_threshold=threshold,
        escalation_callback=callback,
    )

    for _ in range(threshold + 5):
        tracker.record_failure(check_type)

    assert tracker.get_failure_count(check_type) == threshold + 5
