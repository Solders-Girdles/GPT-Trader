"""
Unit tests for RateLimitTracker.

Tests rate limiting logic with sliding time windows.
"""

from datetime import datetime, timedelta

import pytest

from bot_v2.features.live_trade.rate_limit_tracker import RateLimitTracker


class TestBasicRateLimiting:
    """Tests for basic rate limiting functionality."""

    def test_first_request_allowed(self):
        """Test first request is always allowed."""
        tracker = RateLimitTracker()

        allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)

        assert allowed is True

    def test_requests_under_limit_allowed(self):
        """Test requests under limit are allowed."""
        tracker = RateLimitTracker()

        # Make 5 requests with 10/min limit
        for i in range(5):
            allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
            assert allowed is True, f"Request {i+1} should be allowed"

    def test_requests_at_limit_blocked(self):
        """Test requests at exactly the limit are blocked."""
        tracker = RateLimitTracker()

        # Fill up to limit
        for i in range(10):
            allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
            assert allowed is True, f"Request {i+1} should be allowed"

        # Next request should be blocked
        allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)

        assert allowed is False

    def test_requests_over_limit_blocked(self):
        """Test multiple requests over limit are all blocked."""
        tracker = RateLimitTracker()

        # Fill up to limit
        for i in range(10):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Try 3 more requests, all should be blocked
        for i in range(3):
            allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
            assert allowed is False, f"Over-limit request {i+1} should be blocked"


class TestSlidingWindow:
    """Tests for sliding time window behavior."""

    def test_old_entries_cleaned_up(self):
        """Test entries outside window are cleaned up."""
        # Create mock time provider
        current_time = datetime(2025, 1, 1, 12, 0, 0)

        def mock_time():
            return current_time

        tracker = RateLimitTracker(window_minutes=1, time_provider=mock_time)

        # Make 3 requests
        for i in range(3):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Advance time by 2 minutes (outside window)
        current_time = datetime(2025, 1, 1, 12, 2, 0)

        # Request count should be 0 after cleanup
        count = tracker.get_request_count("BTC-USD")
        assert count == 0

    def test_requests_allowed_after_window_expires(self):
        """Test requests allowed again after window expires."""
        current_time = datetime(2025, 1, 1, 12, 0, 0)

        def mock_time():
            return current_time

        tracker = RateLimitTracker(window_minutes=1, time_provider=mock_time)

        # Fill limit
        for i in range(10):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Should be blocked
        allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
        assert allowed is False

        # Advance time past window
        current_time = datetime(2025, 1, 1, 12, 1, 1)

        # Should be allowed again
        allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
        assert allowed is True

    def test_partial_window_expiration(self):
        """Test partial cleanup when some entries expire."""
        current_time = datetime(2025, 1, 1, 12, 0, 0)

        def mock_time():
            return current_time

        tracker = RateLimitTracker(window_minutes=1, time_provider=mock_time)

        # Make 5 requests at T=0
        for i in range(5):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Advance 30 seconds
        current_time = datetime(2025, 1, 1, 12, 0, 30)

        # Make 5 more requests at T=30s
        for i in range(5):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # At limit now
        assert tracker.get_request_count("BTC-USD") == 10

        # Advance to T=70s (first 5 requests should expire)
        current_time = datetime(2025, 1, 1, 12, 1, 10)

        # Should have 5 requests left
        count = tracker.get_request_count("BTC-USD")
        assert count == 5


class TestPerSymbolTracking:
    """Tests for independent per-symbol tracking."""

    def test_different_symbols_tracked_independently(self):
        """Test different symbols don't interfere."""
        tracker = RateLimitTracker()

        # Fill BTC limit
        for i in range(10):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # BTC should be blocked
        allowed_btc = tracker.check_and_record("BTC-USD", limit_per_minute=10)
        assert allowed_btc is False

        # ETH should still be allowed
        allowed_eth = tracker.check_and_record("ETH-USD", limit_per_minute=10)
        assert allowed_eth is True

    def test_multiple_symbols_tracked(self):
        """Test tracking multiple symbols simultaneously."""
        tracker = RateLimitTracker()

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        for symbol in symbols:
            for i in range(5):
                tracker.check_and_record(symbol, limit_per_minute=10)

        # All should have 5 requests
        for symbol in symbols:
            count = tracker.get_request_count(symbol)
            assert count == 5

    def test_different_limits_per_symbol(self):
        """Test different rate limits work independently."""
        tracker = RateLimitTracker()

        # BTC with 5/min limit
        for i in range(5):
            tracker.check_and_record("BTC-USD", limit_per_minute=5)

        # ETH with 10/min limit
        for i in range(10):
            tracker.check_and_record("ETH-USD", limit_per_minute=10)

        # BTC at limit
        allowed_btc = tracker.check_and_record("BTC-USD", limit_per_minute=5)
        assert allowed_btc is False

        # ETH at limit
        allowed_eth = tracker.check_and_record("ETH-USD", limit_per_minute=10)
        assert allowed_eth is False


class TestRequestCounting:
    """Tests for request count tracking."""

    def test_get_request_count_new_symbol(self):
        """Test count is 0 for new symbol."""
        tracker = RateLimitTracker()

        count = tracker.get_request_count("BTC-USD")

        assert count == 0

    def test_get_request_count_after_requests(self):
        """Test count reflects recorded requests."""
        tracker = RateLimitTracker()

        for i in range(7):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        count = tracker.get_request_count("BTC-USD")
        assert count == 7

    def test_get_request_count_cleans_old_entries(self):
        """Test get_request_count triggers cleanup."""
        current_time = datetime(2025, 1, 1, 12, 0, 0)

        def mock_time():
            return current_time

        tracker = RateLimitTracker(window_minutes=1, time_provider=mock_time)

        # Make 5 requests
        for i in range(5):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Advance time past window
        current_time = datetime(2025, 1, 1, 12, 2, 0)

        # Count should trigger cleanup
        count = tracker.get_request_count("BTC-USD")
        assert count == 0


class TestResetFunctionality:
    """Tests for reset operations."""

    def test_reset_clears_symbol_history(self):
        """Test reset clears rate limit for symbol."""
        tracker = RateLimitTracker()

        # Make requests
        for i in range(10):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Reset
        tracker.reset("BTC-USD")

        # Count should be 0
        count = tracker.get_request_count("BTC-USD")
        assert count == 0

    def test_reset_allows_new_requests(self):
        """Test reset allows requests again."""
        tracker = RateLimitTracker()

        # Fill limit
        for i in range(10):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Should be blocked
        allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
        assert allowed is False

        # Reset
        tracker.reset("BTC-USD")

        # Should be allowed now
        allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
        assert allowed is True

    def test_reset_nonexistent_symbol_safe(self):
        """Test resetting non-existent symbol doesn't error."""
        tracker = RateLimitTracker()

        # Should not raise
        tracker.reset("NONEXISTENT-USD")

    def test_reset_all_clears_all_symbols(self):
        """Test reset_all clears all symbols."""
        tracker = RateLimitTracker()

        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

        for symbol in symbols:
            for i in range(5):
                tracker.check_and_record(symbol, limit_per_minute=10)

        # Reset all
        tracker.reset_all()

        # All counts should be 0
        for symbol in symbols:
            count = tracker.get_request_count(symbol)
            assert count == 0


class TestTrackedSymbols:
    """Tests for getting tracked symbols."""

    def test_get_tracked_symbols_empty(self):
        """Test empty tracker returns empty list."""
        tracker = RateLimitTracker()

        symbols = tracker.get_tracked_symbols()

        assert symbols == []

    def test_get_tracked_symbols_after_requests(self):
        """Test tracked symbols includes all recorded symbols."""
        tracker = RateLimitTracker()

        tracker.check_and_record("BTC-USD", limit_per_minute=10)
        tracker.check_and_record("ETH-USD", limit_per_minute=10)
        tracker.check_and_record("SOL-USD", limit_per_minute=10)

        symbols = tracker.get_tracked_symbols()

        assert set(symbols) == {"BTC-USD", "ETH-USD", "SOL-USD"}


class TestTimeUntilNextAllowed:
    """Tests for calculating time until next request allowed."""

    def test_time_until_next_allowed_no_history(self):
        """Test returns None when no history."""
        tracker = RateLimitTracker()

        time_delta = tracker.get_time_until_next_allowed("BTC-USD", limit_per_minute=10)

        assert time_delta is None

    def test_time_until_next_allowed_under_limit(self):
        """Test returns None when under limit."""
        tracker = RateLimitTracker()

        tracker.check_and_record("BTC-USD", limit_per_minute=10)

        time_delta = tracker.get_time_until_next_allowed("BTC-USD", limit_per_minute=10)

        assert time_delta is None

    def test_time_until_next_allowed_at_limit(self):
        """Test returns time delta when at limit."""
        current_time = datetime(2025, 1, 1, 12, 0, 0)

        def mock_time():
            return current_time

        tracker = RateLimitTracker(window_minutes=1, time_provider=mock_time)

        # Fill limit
        for i in range(10):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Get time until next allowed
        time_delta = tracker.get_time_until_next_allowed("BTC-USD", limit_per_minute=10)

        assert time_delta is not None
        assert 59 <= time_delta.total_seconds() <= 61  # ~60 seconds

    def test_time_until_next_allowed_partial_expiry(self):
        """Test time calculation with partial window expiry."""
        current_time = datetime(2025, 1, 1, 12, 0, 0)

        def mock_time():
            return current_time

        tracker = RateLimitTracker(window_minutes=1, time_provider=mock_time)

        # Fill limit
        for i in range(10):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Advance 30 seconds
        current_time = datetime(2025, 1, 1, 12, 0, 30)

        # Time until next should be ~30 seconds
        time_delta = tracker.get_time_until_next_allowed("BTC-USD", limit_per_minute=10)

        assert time_delta is not None
        assert 29 <= time_delta.total_seconds() <= 31


class TestWindowConfiguration:
    """Tests for custom window configuration."""

    def test_custom_window_size(self):
        """Test tracker respects custom window size."""
        current_time = datetime(2025, 1, 1, 12, 0, 0)

        def mock_time():
            return current_time

        # 2-minute window
        tracker = RateLimitTracker(window_minutes=2, time_provider=mock_time)

        # Make requests
        for i in range(5):
            tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Advance 1 minute (within 2-min window)
        current_time = datetime(2025, 1, 1, 12, 1, 0)

        # Should still have 5 requests
        count = tracker.get_request_count("BTC-USD")
        assert count == 5

        # Advance to 2.5 minutes (outside window)
        current_time = datetime(2025, 1, 1, 12, 2, 30)

        # Should have 0 requests
        count = tracker.get_request_count("BTC-USD")
        assert count == 0


class TestTimeProviderInjection:
    """Tests for time provider dependency injection."""

    def test_custom_time_provider(self):
        """Test tracker uses injected time provider."""
        fixed_time = datetime(2025, 6, 15, 10, 30, 0)

        def fixed_time_provider():
            return fixed_time

        tracker = RateLimitTracker(time_provider=fixed_time_provider)

        # Make request
        tracker.check_and_record("BTC-USD", limit_per_minute=10)

        # Internal timestamp should use fixed time
        # (We can't directly inspect, but behavior confirms it)
        count = tracker.get_request_count("BTC-USD")
        assert count == 1

    def test_default_time_provider(self):
        """Test default time provider works."""
        tracker = RateLimitTracker()

        # Should not raise
        allowed = tracker.check_and_record("BTC-USD", limit_per_minute=10)
        assert allowed is True
