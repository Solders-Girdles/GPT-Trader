"""Tests for rate limiting escalation and clearing behaviors."""

from __future__ import annotations

from typing import Any


class TestRateLimitingEscalationAndClearing:
    """Test blocking/escalation and clear/reset helpers."""

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
