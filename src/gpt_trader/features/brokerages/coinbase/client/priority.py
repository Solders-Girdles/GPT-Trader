"""
Request Priority System for API rate limit management.

Ensures critical requests (orders, cancellations) are prioritized
over background requests (market data, products) when approaching
rate limits.
"""

import threading
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="request_priority")


class RequestPriority(IntEnum):
    """Request priority levels (lower number = higher priority)."""

    CRITICAL = 0  # Orders, cancellations - must execute
    HIGH = 1  # Accounts, positions - important for trading decisions
    NORMAL = 2  # Market data - real-time but deferrable
    LOW = 3  # Products, historical data - can wait


# Endpoint patterns mapped to priority levels
ENDPOINT_PRIORITIES: dict[str, RequestPriority] = {
    # Critical - order management
    "orders": RequestPriority.CRITICAL,
    "cancel": RequestPriority.CRITICAL,
    # High - account/position data for trading decisions
    "accounts": RequestPriority.HIGH,
    "positions": RequestPriority.HIGH,
    "cfm/balance": RequestPriority.HIGH,
    "cfm/positions": RequestPriority.HIGH,
    "intx/positions": RequestPriority.HIGH,
    "fills": RequestPriority.HIGH,
    # Normal - market data
    "ticker": RequestPriority.NORMAL,
    "best_bid_ask": RequestPriority.NORMAL,
    "market/": RequestPriority.NORMAL,
    # Low - stable/historical data
    "products": RequestPriority.LOW,
    "candles": RequestPriority.LOW,
}


@dataclass
class PriorityManager:
    """Manages request prioritization based on rate limit pressure.

    When rate limit usage is low, all requests proceed.
    As usage increases, lower priority requests are deferred.

    Thresholds (configurable):
    - Below 70%: All requests allowed
    - 70-85%: Only HIGH and above
    - 85-95%: Only CRITICAL
    - Above 95%: Only CRITICAL (with warning)
    """

    # Usage thresholds for priority enforcement
    threshold_high: float = 0.70  # Start blocking LOW priority
    threshold_critical: float = 0.85  # Start blocking NORMAL priority
    threshold_emergency: float = 0.95  # Warn even for CRITICAL

    enabled: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Stats tracking
    _blocked_requests: dict[str, int] = field(
        default_factory=lambda: {
            "low": 0,
            "normal": 0,
            "high": 0,
        }
    )
    _allowed_requests: int = 0

    def _get_priority(self, path: str) -> RequestPriority:
        """Determine priority for a given endpoint path."""
        path_lower = path.lower()
        for keyword, priority in ENDPOINT_PRIORITIES.items():
            if keyword in path_lower:
                return priority
        return RequestPriority.NORMAL

    def should_allow(self, path: str, rate_limit_usage: float) -> bool:
        """Check if a request should be allowed based on priority and usage.

        Args:
            path: The endpoint path
            rate_limit_usage: Current rate limit usage (0.0 to 1.0+)

        Returns:
            True if the request should proceed, False if it should be deferred.
        """
        if not self.enabled:
            return True

        priority = self._get_priority(path)

        with self._lock:
            # Always allow CRITICAL requests
            if priority == RequestPriority.CRITICAL:
                if rate_limit_usage >= self.threshold_emergency:
                    logger.warning(
                        f"Rate limit at {rate_limit_usage:.0%} - allowing critical request: {path}"
                    )
                self._allowed_requests += 1
                return True

            # Check thresholds for other priorities
            if rate_limit_usage >= self.threshold_critical:
                # Only CRITICAL allowed
                self._blocked_requests[priority.name.lower()] = (
                    self._blocked_requests.get(priority.name.lower(), 0) + 1
                )
                logger.debug(
                    f"Blocking {priority.name} request at {rate_limit_usage:.0%} usage: {path}"
                )
                return False

            if rate_limit_usage >= self.threshold_high:
                # Only HIGH and CRITICAL allowed
                if priority >= RequestPriority.NORMAL:
                    self._blocked_requests[priority.name.lower()] = (
                        self._blocked_requests.get(priority.name.lower(), 0) + 1
                    )
                    logger.debug(
                        f"Blocking {priority.name} request at {rate_limit_usage:.0%} usage: {path}"
                    )
                    return False

            self._allowed_requests += 1
            return True

    def get_priority(self, path: str) -> RequestPriority:
        """Get the priority level for a path (public method)."""
        return self._get_priority(path)

    def get_stats(self) -> dict[str, Any]:
        """Get priority enforcement statistics."""
        with self._lock:
            total_blocked = sum(self._blocked_requests.values())
            total = self._allowed_requests + total_blocked

            return {
                "enabled": self.enabled,
                "allowed_requests": self._allowed_requests,
                "blocked_requests": dict(self._blocked_requests),
                "total_blocked": total_blocked,
                "block_rate": total_blocked / total if total > 0 else 0.0,
                "thresholds": {
                    "high": self.threshold_high,
                    "critical": self.threshold_critical,
                    "emergency": self.threshold_emergency,
                },
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._lock:
            self._blocked_requests = {"low": 0, "normal": 0, "high": 0}
            self._allowed_requests = 0


class RequestDeferredError(Exception):
    """Raised when a request is deferred due to priority throttling."""

    def __init__(self, path: str, priority: RequestPriority, usage: float):
        self.path = path
        self.priority = priority
        self.usage = usage
        super().__init__(
            f"Request deferred: {priority.name} priority at {usage:.0%} rate limit usage"
        )
