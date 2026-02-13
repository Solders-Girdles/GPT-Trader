"""Tests for the shared backoff policy helper."""

from __future__ import annotations

from gpt_trader.utilities.backoff_policy import BackoffDecision, evaluate_backoff_delay


def test_evaluate_backoff_delay_zero_for_first_attempt() -> None:
    decision = evaluate_backoff_delay(
        attempt=1,
        base_delay=0.5,
        max_delay=2.0,
    )

    assert isinstance(decision, BackoffDecision)
    assert decision.delay_seconds == 0.0
    assert decision.capped is False


def test_delay_increases_and_caps_at_max_delay() -> None:
    decision = evaluate_backoff_delay(
        attempt=4,
        base_delay=0.1,
        max_delay=0.3,
        multiplier=2.0,
        jitter=0.0,
    )

    assert decision.delay_seconds == 0.3
    assert decision.capped is True
    assert decision.attempt == 4


def test_jitter_is_applied_via_random_fn() -> None:
    decision = evaluate_backoff_delay(
        attempt=3,
        base_delay=0.1,
        max_delay=1.0,
        multiplier=2.0,
        jitter=0.5,
        random_fn=lambda: 0.2,
    )

    assert decision.delay_seconds > 0.1
