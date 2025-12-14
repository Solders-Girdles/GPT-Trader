"""
API Metrics Collection for observability.

Tracks latency, error rates, and request counts per endpoint
to populate SystemStatus.api_latency and enable monitoring.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="api_metrics")


@dataclass
class EndpointMetrics:
    """Metrics for a single endpoint category."""

    total_calls: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")

    # Recent latencies for percentile calculation
    recent_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    @property
    def average_latency_ms(self) -> float:
        """Average latency across all calls."""
        return self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate as a fraction (0.0 to 1.0)."""
        return self.total_errors / self.total_calls if self.total_calls > 0 else 0.0

    @property
    def p50_latency_ms(self) -> float:
        """50th percentile (median) latency."""
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency."""
        return self._percentile(95)

    @property
    def p99_latency_ms(self) -> float:
        """99th percentile latency."""
        return self._percentile(99)

    def _percentile(self, p: int) -> float:
        """Calculate the p-th percentile of recent latencies."""
        if not self.recent_latencies:
            return 0.0
        sorted_latencies = sorted(self.recent_latencies)
        index = int(len(sorted_latencies) * p / 100)
        index = min(index, len(sorted_latencies) - 1)
        return sorted_latencies[index]

    def record(self, latency_ms: float, error: bool = False) -> None:
        """Record a single request."""
        self.total_calls += 1
        self.total_latency_ms += latency_ms
        self.last_latency_ms = latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.recent_latencies.append(latency_ms)

        if error:
            self.total_errors += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.average_latency_ms, 2),
            "last_latency_ms": round(self.last_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float("inf") else 0.0,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
        }


@dataclass
class APIMetricsCollector:
    """Collects and aggregates API metrics across all endpoints.

    Thread-safe collector that tracks latency, errors, and request counts.
    Used to populate SystemStatus.api_latency and provide observability.
    """

    # Maximum recent requests to keep for overall stats
    max_history: int = 100
    enabled: bool = True

    _endpoint_metrics: dict[str, EndpointMetrics] = field(default_factory=dict)
    _recent_latencies: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _lock: threading.RLock = field(default_factory=threading.RLock)
    _start_time: float = field(default_factory=time.time)

    # Counters for overall stats
    _total_requests: int = 0
    _total_errors: int = 0
    _rate_limit_hits: int = 0

    # Endpoint path keywords -> category
    ENDPOINT_CATEGORIES: dict[str, str] = field(default_factory=lambda: {
        "orders": "orders",
        "fills": "orders",
        "accounts": "accounts",
        "positions": "positions",
        "cfm": "cfm",
        "intx": "intx",
        "products": "products",
        "market": "market",
        "ticker": "market",
        "candles": "market",
        "best_bid_ask": "market",
    })

    def _categorize_endpoint(self, path: str) -> str:
        """Determine the category for an endpoint path."""
        path_lower = path.lower()
        for keyword, category in self.ENDPOINT_CATEGORIES.items():
            if keyword in path_lower:
                return category
        return "other"

    def record_request(
        self,
        path: str,
        latency_ms: float,
        error: bool = False,
        rate_limited: bool = False,
    ) -> None:
        """Record an API request.

        Args:
            path: The endpoint path
            latency_ms: Request latency in milliseconds
            error: Whether the request resulted in an error
            rate_limited: Whether the request was rate limited (429)
        """
        if not self.enabled:
            return

        category = self._categorize_endpoint(path)

        with self._lock:
            # Update endpoint-specific metrics
            if category not in self._endpoint_metrics:
                self._endpoint_metrics[category] = EndpointMetrics()

            self._endpoint_metrics[category].record(latency_ms, error)

            # Update overall stats
            self._recent_latencies.append(latency_ms)
            self._total_requests += 1
            if error:
                self._total_errors += 1
            if rate_limited:
                self._rate_limit_hits += 1

    def get_average_latency(self) -> float:
        """Get overall average latency in milliseconds."""
        with self._lock:
            if not self._recent_latencies:
                return 0.0
            return sum(self._recent_latencies) / len(self._recent_latencies)

    def get_error_rate(self) -> float:
        """Get overall error rate as a fraction."""
        with self._lock:
            if self._total_requests == 0:
                return 0.0
            return self._total_errors / self._total_requests

    def get_rate_limit_usage_display(self, current_usage: float) -> str:
        """Format rate limit usage for display.

        Args:
            current_usage: Current usage as fraction (0.0 to 1.0)

        Returns:
            Formatted string like "45/100 (45%)"
        """
        # This is a display helper - actual usage comes from rate limiter
        percentage = int(current_usage * 100)
        return f"{percentage}%"

    def get_endpoint_metrics(self, category: str) -> EndpointMetrics | None:
        """Get metrics for a specific endpoint category."""
        with self._lock:
            return self._endpoint_metrics.get(category)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all metrics.

        Returns:
            Dict suitable for status reporting and monitoring.
        """
        with self._lock:
            uptime = time.time() - self._start_time

            # Calculate overall percentiles
            sorted_latencies = sorted(self._recent_latencies) if self._recent_latencies else []

            def percentile(p: int) -> float:
                if not sorted_latencies:
                    return 0.0
                index = min(int(len(sorted_latencies) * p / 100), len(sorted_latencies) - 1)
                return sorted_latencies[index]

            return {
                "uptime_seconds": round(uptime, 1),
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "error_rate": round(self.get_error_rate(), 4),
                "rate_limit_hits": self._rate_limit_hits,
                "avg_latency_ms": round(self.get_average_latency(), 2),
                "p50_latency_ms": round(percentile(50), 2),
                "p95_latency_ms": round(percentile(95), 2),
                "p99_latency_ms": round(percentile(99), 2),
                "endpoints": {
                    category: metrics.to_dict()
                    for category, metrics in self._endpoint_metrics.items()
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._endpoint_metrics.clear()
            self._recent_latencies.clear()
            self._total_requests = 0
            self._total_errors = 0
            self._rate_limit_hits = 0
            self._start_time = time.time()


class RequestTimer:
    """Context manager for timing requests.

    Usage:
        with RequestTimer(metrics, "/api/orders") as timer:
            response = client.get("/api/orders")
            if response.status_code >= 400:
                timer.mark_error()
    """

    def __init__(self, collector: APIMetricsCollector, path: str):
        self.collector = collector
        self.path = path
        self.start_time: float = 0.0
        self.error = False
        self.rate_limited = False

    def __enter__(self) -> "RequestTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.collector.record_request(
            self.path,
            elapsed_ms,
            error=self.error or exc_type is not None,
            rate_limited=self.rate_limited,
        )

    def mark_error(self) -> None:
        """Mark this request as an error."""
        self.error = True

    def mark_rate_limited(self) -> None:
        """Mark this request as rate limited."""
        self.rate_limited = True
        self.error = True
