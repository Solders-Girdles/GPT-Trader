"""
Rate Limit Tracker for Order Throttling.

Tracks order submission rate limits per symbol using a sliding time window.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimitTracker:
    """
    Tracks rate limits per symbol using sliding time window.

    Maintains timestamp history for each symbol and enforces
    per-minute rate limits to prevent exchange throttling.
    """

    def __init__(
        self,
        window_minutes: int = 1,
        time_provider: Callable[[], datetime] | None = None,
    ) -> None:
        """
        Initialize rate limit tracker.

        Args:
            window_minutes: Size of sliding time window for rate limiting
            time_provider: Optional callable that returns current time (for testing)
        """
        self.window_minutes = window_minutes
        self._time_provider = time_provider or datetime.now
        self._rate_limits: dict[str, list[datetime]] = {}

        logger.debug(f"RateLimitTracker initialized with {window_minutes}min window")

    def check_and_record(self, symbol: str, limit_per_minute: int) -> bool:
        """
        Check if symbol is within rate limit and record request if allowed.

        Uses sliding window to track requests. Automatically cleans up
        old timestamps outside the window.

        Args:
            symbol: Trading symbol
            limit_per_minute: Maximum requests allowed per minute

        Returns:
            True if request is allowed (and recorded), False if rate limit exceeded
        """
        now = self._time_provider()

        # Initialize if needed
        if symbol not in self._rate_limits:
            self._rate_limits[symbol] = []

        # Clean old entries outside window
        self._cleanup_old_entries(symbol, now)

        # Check limit
        current_count = len(self._rate_limits[symbol])
        if current_count >= limit_per_minute:
            logger.warning(
                f"Rate limit exceeded for {symbol}: {current_count}/{limit_per_minute} requests"
            )
            return False

        # Record request
        self._rate_limits[symbol].append(now)
        logger.debug(f"Recorded request for {symbol}: {current_count + 1}/{limit_per_minute}")

        return True

    def get_request_count(self, symbol: str) -> int:
        """
        Get current request count in window.

        Args:
            symbol: Trading symbol

        Returns:
            Number of requests in current time window
        """
        if symbol not in self._rate_limits:
            return 0

        # Clean old entries first
        now = self._time_provider()
        self._cleanup_old_entries(symbol, now)

        return len(self._rate_limits[symbol])

    def reset(self, symbol: str) -> None:
        """
        Reset rate limit tracking for symbol.

        Args:
            symbol: Trading symbol to reset
        """
        if symbol in self._rate_limits:
            self._rate_limits[symbol] = []
            logger.debug(f"Reset rate limit for {symbol}")

    def reset_all(self) -> None:
        """Reset all rate limit tracking."""
        self._rate_limits.clear()
        logger.debug("Reset all rate limits")

    def get_tracked_symbols(self) -> list[str]:
        """
        Get list of symbols currently being tracked.

        Returns:
            List of symbol names
        """
        return list(self._rate_limits.keys())

    def get_time_until_next_allowed(self, symbol: str, limit_per_minute: int) -> timedelta | None:
        """
        Calculate time until next request would be allowed.

        Args:
            symbol: Trading symbol
            limit_per_minute: Rate limit threshold

        Returns:
            Time delta until next request allowed, or None if requests currently allowed
        """
        if symbol not in self._rate_limits:
            return None  # No history, requests allowed

        now = self._time_provider()
        self._cleanup_old_entries(symbol, now)

        current_count = len(self._rate_limits[symbol])
        if current_count < limit_per_minute:
            return None  # Under limit, requests allowed

        # Find oldest timestamp in window
        if not self._rate_limits[symbol]:
            return None

        oldest_timestamp = min(self._rate_limits[symbol])
        window_end = oldest_timestamp + timedelta(minutes=self.window_minutes)

        time_until_allowed = window_end - now
        return time_until_allowed if time_until_allowed.total_seconds() > 0 else None

    def _cleanup_old_entries(self, symbol: str, now: datetime) -> None:
        """
        Remove timestamps outside the sliding window.

        Args:
            symbol: Trading symbol
            now: Current time
        """
        if symbol not in self._rate_limits:
            return

        cutoff = now - timedelta(minutes=self.window_minutes)
        original_count = len(self._rate_limits[symbol])

        # Keep only timestamps within window
        self._rate_limits[symbol] = [ts for ts in self._rate_limits[symbol] if ts >= cutoff]

        cleaned_count = original_count - len(self._rate_limits[symbol])
        if cleaned_count > 0:
            logger.debug(f"Cleaned {cleaned_count} old entries for {symbol}")
