"""Tests for the failure injection harness.

Validates the harness components used for resilience testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.fixtures.failure_injection import (
    FailureScript,
    InjectingBroker,
    counting_sleep,
    no_op_sleep,
)


class TestFailureScriptFromOutcomes:
    """Tests for FailureScript.from_outcomes() class method."""

    def test_from_outcomes_builds_from_varargs(self) -> None:
        """Test that from_outcomes() builds from variable arguments."""
        exc1 = ConnectionError("fail 1")
        exc2 = TimeoutError("fail 2")
        script = FailureScript.from_outcomes(exc1, exc2, None)

        assert len(script.sequence) == 3
        assert script.sequence[0] is exc1
        assert script.sequence[1] is exc2
        assert script.sequence[2] is None

    def test_from_outcomes_with_loop(self) -> None:
        """Test that from_outcomes() respects loop parameter."""
        script = FailureScript.from_outcomes(None, loop=True)
        assert script.loop is True

    def test_from_outcomes_empty(self) -> None:
        """Test empty sequence."""
        script = FailureScript.from_outcomes()
        assert len(script.sequence) == 0


class TestFailureScriptCallCount:
    """Tests for FailureScript.call_count property."""

    def test_call_count_starts_at_zero(self) -> None:
        """Test that call_count starts at zero."""
        script = FailureScript.fail_then_succeed(3)
        assert script.call_count == 0

    def test_call_count_increments(self) -> None:
        """Test that call_count increments with each outcome."""
        script = FailureScript.from_outcomes(
            ConnectionError("fail"),
            None,  # success
        )

        assert script.call_count == 0

        try:
            script.next_outcome()
        except Exception:
            pass
        assert script.call_count == 1

        script.next_outcome()
        assert script.call_count == 2

    def test_call_count_reset(self) -> None:
        """Test that reset() clears call_count."""
        script = FailureScript.fail_then_succeed(2)

        script.next_outcome()
        script.next_outcome()
        assert script.call_count == 2

        script.reset()
        assert script.call_count == 0


class TestFailureScriptExhaustion:
    """Tests for FailureScript exhaustion behavior."""

    def test_exhausted_raises_stop_iteration(self) -> None:
        """Test that exhausted script raises StopIteration."""
        script = FailureScript.from_outcomes(None)  # single success

        script.next_outcome()  # consumes the success

        with pytest.raises(StopIteration, match="exhausted"):
            script.next_outcome()

    def test_loop_prevents_exhaustion(self) -> None:
        """Test that loop=True prevents exhaustion."""
        exc = ConnectionError("always fail")
        script = FailureScript.from_outcomes(exc, loop=True)

        # Should not exhaust
        for _ in range(10):
            outcome = script.next_outcome()
            assert outcome is exc


class TestInjectingBrokerCallCount:
    """Tests for InjectingBroker call counting."""

    def test_get_call_count_tracks_calls(self) -> None:
        """Test that get_call_count tracks method calls."""
        mock_broker = MagicMock()
        mock_broker.place_order.return_value = "order"
        script = FailureScript.fail_then_succeed(2)
        injecting = InjectingBroker(mock_broker, place_order=script)

        assert injecting.get_call_count("place_order") == 0

        try:
            injecting.place_order()
        except Exception:
            pass
        assert injecting.get_call_count("place_order") == 1

        try:
            injecting.place_order()
        except Exception:
            pass
        assert injecting.get_call_count("place_order") == 2

        injecting.place_order()  # success
        assert injecting.get_call_count("place_order") == 3

    def test_get_call_count_zero_for_uncalled(self) -> None:
        """Test that get_call_count returns 0 for uncalled methods."""
        mock_broker = MagicMock()
        injecting = InjectingBroker(mock_broker)

        assert injecting.get_call_count("place_order") == 0


class TestCountingSleep:
    """Tests for counting_sleep helper."""

    def test_counting_sleep_records_durations(self) -> None:
        """Test that counting_sleep records sleep durations."""
        sleep_fn, get_sleeps = counting_sleep()

        sleep_fn(0.5)
        sleep_fn(1.0)
        sleep_fn(2.0)

        sleeps = get_sleeps()
        assert sleeps == [0.5, 1.0, 2.0]

    def test_counting_sleep_returns_copy(self) -> None:
        """Test that get_sleeps returns a copy."""
        sleep_fn, get_sleeps = counting_sleep()

        sleep_fn(1.0)
        sleeps1 = get_sleeps()
        sleeps1.append(999)  # modify the copy

        sleeps2 = get_sleeps()
        assert 999 not in sleeps2


class TestNoOpSleep:
    """Tests for no_op_sleep helper."""

    def test_no_op_sleep_does_nothing(self) -> None:
        """Test that no_op_sleep is callable and returns None."""
        result = no_op_sleep(100)  # large value should be instant
        assert result is None
