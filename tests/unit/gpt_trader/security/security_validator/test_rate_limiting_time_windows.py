"""Tests for time-based rate limiting in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestRateLimitingTimeWindows:
    """Test rate limiting behaviors that depend on time windows."""

    def test_rate_limit_time_window_reset(self, security_validator: Any, frozen_time: Any) -> None:
        """Test rate limit resets after time window."""
        user_id = "test-user"
        limit_type = "api_calls"

        # Exhaust limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        period = security_validator.RATE_LIMITS[limit_type].period

        for i in range(limit + 1):
            security_validator.check_rate_limit(user_id, limit_type)

        # Should be blocked
        allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False

        # Advance time beyond period
        frozen_time.tick(delta=period + 1)

        # Should be allowed again
        allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is True

    def test_rate_limit_time_precision(self, security_validator: Any, frozen_time: Any) -> None:
        """Test rate limiting time precision."""
        user_id = "test-user"
        limit_type = "api_calls"

        # Make requests at same timestamp
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit):
            allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
            assert allowed is True

        # Should be blocked at same timestamp
        allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False

        # Advance time slightly (but not enough to reset)
        frozen_time.tick(delta=1)

        # Should still be blocked
        allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False
