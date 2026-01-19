"""Tests for concurrency behavior in SecurityValidator rate limiting."""

from __future__ import annotations

import threading
from typing import Any


class TestRateLimitingConcurrency:
    """Test rate limiting under concurrent access."""

    def test_rate_limit_concurrent_requests(self, security_validator: Any) -> None:
        """Test rate limiting with concurrent requests."""
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
