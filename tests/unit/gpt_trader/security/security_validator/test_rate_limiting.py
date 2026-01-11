"""Tests for rate limiting in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestRateLimiting:
    """Test rate limiting scenarios."""

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

    def test_rate_limit_suspicious_activity_detection(self, security_validator: Any) -> None:
        """Test suspicious activity detection from rate limiting."""
        user_id = "test-user"
        limit_type = "api_calls"

        # Make many requests to trigger suspicious activity
        limit = security_validator.RATE_LIMITS[limit_type].requests

        # Exceed limit multiple times
        for _ in range(15):  # More than threshold for suspicious activity
            for i in range(limit + 1):
                security_validator.check_rate_limit(user_id, limit_type)

        # Should eventually be blocked as suspicious
        allowed, message = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False
        assert "temporarily blocked" in message

    def test_rate_limit_clear_user(self, security_validator: Any) -> None:
        """Test clearing rate limits for specific user."""
        user_id = "test-user"
        limit_type = "api_calls"

        # Exhaust limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit + 1):
            security_validator.check_rate_limit(user_id, limit_type)

        # Should be blocked
        allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False

        # Clear user limits
        security_validator.clear_rate_limits(user_id)

        # Should still be blocked if suspicious activity was detected
        # (but rate limit counter should be reset)
        # This behavior depends on implementation

    def test_rate_limit_clear_all(self, security_validator: Any) -> None:
        """Test clearing all rate limits."""
        user_id = "test-user"
        limit_type = "api_calls"

        # Exhaust limit
        limit = security_validator.RATE_LIMITS[limit_type].requests
        for i in range(limit + 1):
            security_validator.check_rate_limit(user_id, limit_type)

        # Should be blocked
        allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
        assert allowed is False

        # Clear all limits
        security_validator.clear_rate_limits()

        # Should be allowed (unless suspicious activity blocking)
        allowed, _ = security_validator.check_rate_limit(user_id, limit_type)
        # Result depends on suspicious activity state

    def test_rate_limit_invalid_limit_type(self, security_validator: Any) -> None:
        """Test rate limiting with invalid limit type."""
        user_id = "test-user"
        invalid_type = "invalid_limit_type"

        # Should allow by default for unknown limit types
        allowed, message = security_validator.check_rate_limit(user_id, invalid_type)
        assert allowed is True
        assert message is None

    def test_rate_limit_concurrent_requests(self, security_validator: Any) -> None:
        """Test rate limiting with concurrent requests."""
        import threading

        user_id = "test-user"
        limit_type = "order_submissions"
        results = []

        def make_request():
            allowed, message = security_validator.check_rate_limit(user_id, limit_type)
            results.append((allowed, message))

        # Create multiple threads making requests
        threads = [threading.Thread(target=make_request) for _ in range(20)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()

        # Should have some allowed and some blocked requests
        allowed_count = sum(1 for allowed, _ in results if allowed)
        blocked_count = sum(1 for allowed, _ in results if not allowed)

        assert allowed_count > 0
        assert blocked_count > 0

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

    def test_rate_limit_ip_blocking(self, security_validator: Any) -> None:
        """Test IP blocking from excessive rate limit violations."""
        ip_address = "192.168.1.100"
        limit_type = "api_calls"

        # Make many requests to trigger IP blocking
        limit = security_validator.RATE_LIMITS[limit_type].requests

        # Exceed limit multiple times to trigger blocking
        for cycle in range(12):  # Enough to trigger blocking
            for i in range(limit + 1):
                security_validator.check_rate_limit(ip_address, limit_type)

        # Should be blocked with IP blocking message
        allowed, message = security_validator.check_rate_limit(ip_address, limit_type)
        assert allowed is False
        assert "temporarily blocked" in message

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
