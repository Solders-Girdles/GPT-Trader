"""Rate limiting functionality for API and trading operations."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="security")


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""

    requests: int
    period: int  # seconds
    burst: int | None = None


class RateLimiter:
    """Rate limiter with configurable limits and blocking."""

    # Rate limit configurations
    RATE_LIMITS = {
        "api_calls": RateLimitConfig(100, 60),  # 100/minute
        "order_submissions": RateLimitConfig(10, 60, burst=3),  # 10/minute with burst
        "login_attempts": RateLimitConfig(5, 3600),  # 5/hour
        "data_requests": RateLimitConfig(1000, 3600),  # 1000/hour
    }

    def __init__(self) -> None:
        self._lock = Lock()
        self._rate_limiters: defaultdict[str, defaultdict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self._blocked_ips: set[str] = set()
        self._suspicious_activity: defaultdict[str, int] = defaultdict(int)

    def check_rate_limit(self, identifier: str, limit_type: str) -> tuple[bool, str | None]:
        """
        Check if request is within rate limits.

        Args:
            identifier: User ID or IP address
            limit_type: Type of rate limit to check

        Returns:
            Tuple of (allowed, error_message)
        """
        if identifier in self._blocked_ips:
            return False, "IP temporarily blocked due to suspicious activity"

        if limit_type not in self.RATE_LIMITS:
            return True, None

        config = self.RATE_LIMITS[limit_type]
        now = time.time()

        with self._lock:
            # Get request history
            history = self._rate_limiters[limit_type][identifier]

            # Remove old entries
            cutoff = now - config.period
            while history and history[0] < cutoff:
                history.popleft()

            # Check limit
            if len(history) >= config.requests:
                # Check for suspicious activity
                self._suspicious_activity[identifier] += 1
                if self._suspicious_activity[identifier] > 10:
                    self._blocked_ips.add(identifier)
                    logger.warning(
                        f"Blocked {identifier} for excessive rate limit violations",
                        operation="rate_limit",
                        status="blocked",
                    )

                return (
                    False,
                    f"Rate limit exceeded: {config.requests} requests per {config.period} seconds",
                )

            # Add current request
            history.append(now)

            return True, None

    def clear_rate_limits(self, identifier: str | None = None) -> None:
        """Clear rate limit history"""
        with self._lock:
            if identifier:
                for limit_type in self._rate_limiters:
                    if identifier in self._rate_limiters[limit_type]:
                        del self._rate_limiters[limit_type][identifier]
            else:
                self._rate_limiters.clear()
