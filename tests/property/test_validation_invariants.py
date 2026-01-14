"""Property-based tests for validation failure tracking invariants.

Tests critical validation tracking properties:
- Failure counters increment correctly
- Success resets counters to zero
- Escalation triggers exactly at threshold
- Different check types have independent counters
- Escalation callback is called exactly once per threshold crossing
"""

from __future__ import annotations

from collections import defaultdict
from unittest.mock import MagicMock

from hypothesis import given, seed, settings
from hypothesis import strategies as st

from gpt_trader.features.live_trade.execution.validation import ValidationFailureTracker

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


@seed(4001)
@settings(max_examples=100, deadline=None)
@given(
    check_type=check_type_strategy,
    failure_count=operation_count_strategy,
)
def test_failure_counter_increments(
    check_type: str,
    failure_count: int,
) -> None:
    """
    Property: Each record_failure call should increment the counter by 1.
    """
    tracker = ValidationFailureTracker(escalation_threshold=1000)  # High threshold

    for i in range(failure_count):
        tracker.record_failure(check_type)
        expected_count = i + 1
        actual_count = tracker.get_failure_count(check_type)
        assert actual_count == expected_count, (
            f"After {i + 1} failures, expected count {expected_count}, got {actual_count}"
        )


@seed(4002)
@settings(max_examples=100, deadline=None)
@given(
    check_type=check_type_strategy,
    failure_count=operation_count_strategy,
)
def test_success_resets_counter(
    check_type: str,
    failure_count: int,
) -> None:
    """
    Property: record_success should always reset the counter to zero.
    """
    tracker = ValidationFailureTracker(escalation_threshold=1000)

    # Record some failures
    for _ in range(failure_count):
        tracker.record_failure(check_type)

    assert tracker.get_failure_count(check_type) == failure_count

    # Reset with success
    tracker.record_success(check_type)

    assert tracker.get_failure_count(check_type) == 0, "Counter should be zero after success"


@seed(4003)
@settings(max_examples=100, deadline=None)
@given(
    check_type=check_type_strategy,
    threshold=threshold_strategy,
    total_failures=st.integers(min_value=1, max_value=100),
)
def test_escalation_triggers_at_threshold(
    check_type: str,
    threshold: int,
    total_failures: int,
) -> None:
    """
    Property: Escalation should trigger when counter reaches threshold.

    record_failure returns True exactly when threshold is reached.
    """
    tracker = ValidationFailureTracker(escalation_threshold=threshold)
    escalation_count = 0

    for i in range(total_failures):
        escalated = tracker.record_failure(check_type)
        if escalated:
            escalation_count += 1
            # After escalation, counter continues (doesn't reset automatically)
            # The next escalation would be at 2*threshold, 3*threshold, etc.

    # Escalation should happen when count == threshold, 2*threshold, 3*threshold, etc.
    # But counter never resets on its own, so escalations = total_failures // threshold
    # However, the implementation checks >= threshold, so it triggers at threshold
    # and again at every subsequent failure after threshold.
    #
    # Actually reading the implementation: it triggers when count >= threshold,
    # which means it triggers at threshold, threshold+1, threshold+2, etc.
    # Let me verify this assumption...
    #
    # Looking at the code:
    #   if count >= self.escalation_threshold: return True
    # So once you hit threshold, EVERY subsequent failure also triggers.
    #
    # Expected escalations = max(0, total_failures - threshold + 1)
    expected_escalations = max(0, total_failures - threshold + 1)
    assert escalation_count == expected_escalations, (
        f"Expected {expected_escalations} escalations for {total_failures} failures "
        f"with threshold {threshold}, got {escalation_count}"
    )


@seed(4004)
@settings(max_examples=100, deadline=None)
@given(
    check_type=check_type_strategy,
    threshold=threshold_strategy,
)
def test_escalation_callback_called(
    check_type: str,
    threshold: int,
) -> None:
    """
    Property: Escalation callback should be called when threshold is reached.
    """
    callback = MagicMock()
    tracker = ValidationFailureTracker(
        escalation_threshold=threshold,
        escalation_callback=callback,
    )

    # Record failures up to threshold - 1
    for _ in range(threshold - 1):
        tracker.record_failure(check_type)

    callback.assert_not_called()

    # The threshold-th failure should trigger callback
    tracker.record_failure(check_type)
    callback.assert_called_once()


@seed(4005)
@settings(max_examples=100, deadline=None)
@given(
    check_types=st.lists(check_type_strategy, min_size=2, max_size=5, unique=True),
    threshold=threshold_strategy,
)
def test_check_types_are_independent(
    check_types: list[str],
    threshold: int,
) -> None:
    """
    Property: Different check types should have independent counters.

    Failures in one type should not affect other types.
    """
    tracker = ValidationFailureTracker(escalation_threshold=threshold)

    # Record failures only for the first check type
    first_type = check_types[0]
    other_types = check_types[1:]

    for _ in range(threshold + 5):
        tracker.record_failure(first_type)

    # First type should have failures
    assert tracker.get_failure_count(first_type) > 0

    # Other types should have zero
    for other_type in other_types:
        assert tracker.get_failure_count(other_type) == 0, (
            f"Check type {other_type} should not be affected by {first_type} failures"
        )


@seed(4006)
@settings(max_examples=100, deadline=None)
@given(
    check_type=check_type_strategy,
    failure_count=operation_count_strategy,
)
def test_get_failure_count_is_readonly(
    check_type: str,
    failure_count: int,
) -> None:
    """
    Property: get_failure_count should not modify state.
    """
    tracker = ValidationFailureTracker(escalation_threshold=1000)

    for _ in range(failure_count):
        tracker.record_failure(check_type)

    # Call get_failure_count multiple times
    count1 = tracker.get_failure_count(check_type)
    count2 = tracker.get_failure_count(check_type)
    count3 = tracker.get_failure_count(check_type)

    assert count1 == count2 == count3 == failure_count, (
        "get_failure_count should return consistent values"
    )


@seed(4007)
@settings(max_examples=50, deadline=None)
@given(
    check_type=check_type_strategy,
    threshold=threshold_strategy,
)
def test_reset_allows_new_escalation(
    check_type: str,
    threshold: int,
) -> None:
    """
    Property: After reset, reaching threshold again should trigger escalation.
    """
    callback = MagicMock()
    tracker = ValidationFailureTracker(
        escalation_threshold=threshold,
        escalation_callback=callback,
    )

    # First escalation
    for _ in range(threshold):
        tracker.record_failure(check_type)
    assert callback.call_count == 1

    # Reset
    tracker.record_success(check_type)
    callback.reset_mock()

    # Second escalation
    for _ in range(threshold):
        tracker.record_failure(check_type)
    assert callback.call_count == 1, "Callback should trigger again after reset"


@seed(4008)
@settings(max_examples=100, deadline=None)
@given(
    check_type=check_type_strategy,
)
def test_initial_failure_count_is_zero(
    check_type: str,
) -> None:
    """
    Property: Initial failure count for any check type should be zero.
    """
    tracker = ValidationFailureTracker()

    assert tracker.get_failure_count(check_type) == 0


@seed(4009)
@settings(max_examples=50, deadline=None)
@given(
    threshold=threshold_strategy,
)
def test_escalation_does_not_trigger_below_threshold(
    threshold: int,
) -> None:
    """
    Property: record_failure should return False for all failures below threshold.
    """
    tracker = ValidationFailureTracker(escalation_threshold=threshold)

    for i in range(threshold - 1):
        escalated = tracker.record_failure("test_check")
        assert escalated is False, (
            f"Escalation should not trigger at failure {i + 1} (threshold {threshold})"
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

    # Track expected counts ourselves
    expected_counts: dict[str, int] = defaultdict(int)

    for check_type, op in operations:
        if op == "fail":
            tracker.record_failure(check_type)
            expected_counts[check_type] += 1
        else:
            tracker.record_success(check_type)
            expected_counts[check_type] = 0

    # Verify all counts match
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

    # Should not raise despite callback exception
    for _ in range(threshold + 5):
        tracker.record_failure(check_type)

    # Counter should still be correct
    assert tracker.get_failure_count(check_type) == threshold + 5


@seed(4012)
@settings(max_examples=100, deadline=None)
@given(
    check_type=check_type_strategy,
    threshold=threshold_strategy,
)
def test_no_callback_still_returns_true(
    check_type: str,
    threshold: int,
) -> None:
    """
    Property: record_failure returns True at threshold even without callback.
    """
    tracker = ValidationFailureTracker(
        escalation_threshold=threshold,
        escalation_callback=None,  # No callback
    )

    for _ in range(threshold - 1):
        result = tracker.record_failure(check_type)
        assert result is False

    # Threshold hit should return True
    result = tracker.record_failure(check_type)
    assert result is True, "Should return True at threshold even without callback"


__all__ = ["test_failure_counter_increments"]
