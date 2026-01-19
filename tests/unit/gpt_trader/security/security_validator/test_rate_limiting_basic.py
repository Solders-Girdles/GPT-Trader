"""Tests for rate limiting basics in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestRateLimitingBasic:
    """Test basic rate limiting scenarios."""

    def test_rate_limit_within_threshold(self, security_validator: Any) -> None:
        """Test requests within rate limit threshold are allowed."""
        user_id = "test-user"
        limit_type = "api_calls"

        # Make requests up to limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit - 1):
            allowed, message = security_validator.check_rate_limit(user_id, limit_type)
            assert allowed is True
            assert message is None

    def test_rate_limit_exceeds_threshold(self, security_validator: Any) -> None:
        """Test requests exceeding rate limit are blocked."""
        user_id = "test-user"
        limit_type = "api_calls"

        # Make requests up to and beyond limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit):
            allowed, message = security_validator.check_rate_limit(user_id, limit_type)
            assert allowed is True

        # Next request should be blocked
        allowed, message = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False
        assert "Rate limit exceeded" in message

    def test_rate_limit_different_users(self, security_validator: Any) -> None:
        """Test rate limits are applied per user."""
        user1 = "user1"
        user2 = "user2"
        limit_type = "api_calls"

        # User1 exceeds limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit + 1):
            security_validator.check_rate_limit(user1, limit_type)

        # User1 should be blocked
        allowed, _ = security_validator.check_rate_limit(user1, limit_type)
        assert allowed is False

        # User2 should still be allowed
        allowed, _ = security_validator.check_rate_limit(user2, limit_type)
        assert allowed is True

    def test_rate_limit_different_limit_types(self, security_validator: Any) -> None:
        """Test different limit types have independent quotas."""
        user_id = "test-user"

        # Exhaust api_calls limit
        api_limit = security_validator.RATE_LIMITS["api_calls"].requests
        for i in range(api_limit + 1):
            security_validator.check_rate_limit(user_id, "api_calls")

        # Should be blocked for api_calls
        allowed, _ = security_validator.check_rate_limit(user_id, "api_calls")
        assert allowed is False

        # Should still be allowed for order_submissions
        allowed, _ = security_validator.check_rate_limit(user_id, "order_submissions")
        assert allowed is True

    def test_rate_limit_burst_handling(self, security_validator: Any) -> None:
        """Test burst limit handling if configured."""
        user_id = "test-user"
        limit_type = "order_submissions"  # Has burst configured

        # Make rapid requests within burst
        config = security_validator.RATE_LIMITS[limit_type]
        if config.burst:
            for i in range(config.burst):
                allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
                assert allowed is True

    def test_rate_limit_invalid_limit_type(self, security_validator: Any) -> None:
        """Test rate limiting with invalid limit type."""
        user_id = "test-user"
        invalid_type = "invalid_limit_type"

        # Should allow by default for unknown limit types
        allowed, message = security_validator.check_rate_limit(user_id, invalid_type)
        assert allowed is True
        assert message is None

    def test_rate_limit_different_periods(self, security_validator: Any) -> None:
        """Test rate limits with different periods."""
        user_id = "test-user"

        # Test api_calls (60 second period)
        api_config = security_validator.RATE_LIMITS["api_calls"]
        for i in range(api_config.requests):
            allowed, _ = security_validator.check_rate_limit(user_id, "api_calls")
            assert allowed is True

        # Test login_attempts (3600 second period)
        login_config = security_validator.RATE_LIMITS["login_attempts"]
        for i in range(login_config.requests):
            allowed, _ = security_validator.check_rate_limit(user_id, "login_attempts")
            assert allowed is True

    def test_rate_limit_edge_cases(self, security_validator: Any) -> None:
        """Test rate limiting edge cases."""
        allowed_empty, _ = security_validator.check_rate_limit("", "api_calls")
        assert isinstance(allowed_empty, bool)

        allowed_none, _ = security_validator.check_rate_limit(None, "api_calls")  # type: ignore
        assert isinstance(allowed_none, bool)

    def test_rate_limit_order_submissions(self, security_validator: Any) -> None:
        """Test rate limiting for order submissions."""
        user_id = "test-user"
        limit_type = "order_submissions"

        # Make requests up to limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit):
            allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
            assert allowed is True

        # Next request should be blocked
        allowed, message = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False
        assert "Rate limit exceeded" in message

    def test_rate_limit_login_attempts(self, security_validator: Any) -> None:
        """Test rate limiting for login attempts."""
        user_id = "test-user"
        limit_type = "login_attempts"

        # Make requests up to limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit):
            allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
            assert allowed is True

        # Next request should be blocked
        allowed, message = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False
        assert "Rate limit exceeded" in message
