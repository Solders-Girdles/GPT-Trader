"""Tests for RetryPolicy configuration."""

from __future__ import annotations

from gpt_trader.features.live_trade.execution.broker_executor import RetryPolicy


class TestRetryPolicy:
    """Tests for RetryPolicy configuration."""

    def test_default_values(self) -> None:
        """Test default retry policy values."""
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.base_delay == 0.5
        assert policy.max_delay == 5.0
        assert policy.timeout_seconds == 30.0
        assert policy.jitter == 0.1

    def test_calculate_delay_first_attempt_is_zero(self) -> None:
        """Test that first attempt has no delay."""
        policy = RetryPolicy(jitter=0)
        assert policy.calculate_delay(1) == 0.0

    def test_calculate_delay_exponential_backoff(self) -> None:
        """Test exponential backoff calculation."""
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=0)
        assert policy.calculate_delay(2) == 1.0  # base_delay * 2^0
        assert policy.calculate_delay(3) == 2.0  # base_delay * 2^1
        assert policy.calculate_delay(4) == 4.0  # base_delay * 2^2

    def test_calculate_delay_respects_max_delay(self) -> None:
        """Test that delay is capped at max_delay."""
        policy = RetryPolicy(base_delay=1.0, max_delay=2.0, jitter=0)
        assert policy.calculate_delay(10) == 2.0  # Capped at max

    def test_jitter_clamped_negative_to_zero(self) -> None:
        """Test that negative jitter is clamped to 0.0."""
        policy = RetryPolicy(jitter=-0.5)
        assert policy.jitter == 0.0

    def test_jitter_clamped_above_one_to_one(self) -> None:
        """Test that jitter > 1.0 is clamped to 1.0."""
        policy = RetryPolicy(jitter=1.5)
        assert policy.jitter == 1.0

    def test_jitter_valid_range_unchanged(self) -> None:
        """Test that jitter in [0.0, 1.0] is preserved."""
        for jitter_val in [0.0, 0.1, 0.5, 1.0]:
            policy = RetryPolicy(jitter=jitter_val)
            assert policy.jitter == jitter_val

    def test_jitter_boundary_values(self) -> None:
        """Test jitter at exact boundaries."""
        policy_zero = RetryPolicy(jitter=0.0)
        policy_one = RetryPolicy(jitter=1.0)
        assert policy_zero.jitter == 0.0
        assert policy_one.jitter == 1.0
