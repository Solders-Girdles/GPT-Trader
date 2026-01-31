"""
Execution telemetry collection for live trade order submissions.

Tracks order submission latency, success rates, rejection reasons, and retry activity.
Designed for lightweight in-memory aggregation consumed by runtime monitoring and the TUI.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field

# Rolling window size for metrics
METRICS_WINDOW_SIZE = 100
LATENCY_WINDOW_SIZE = 50
ISSUES_WINDOW_SIZE = 10


@dataclass
class ExecutionIssue:
    """A recent execution issue such as a rejection or retry."""

    timestamp: float
    symbol: str
    side: str
    quantity: float
    price: float
    reason: str
    reason_detail: str = ""
    is_retry: bool = False


@dataclass
class ExecutionMetrics:
    """Execution telemetry for order submission tracking.

    Tracks order submission performance including latency, success rates,
    and retry activity for monitoring execution health.
    """

    # Submission counts (rolling window)
    submissions_total: int = 0
    submissions_success: int = 0
    submissions_failed: int = 0
    submissions_rejected: int = 0

    # Latency metrics (milliseconds)
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_latency_ms: float = 0.0

    # Retry metrics
    retry_total: int = 0
    retry_rate: float = 0.0  # retries per submission

    # Recent activity
    last_submission_time: float = 0.0
    last_failure_reason: str = ""

    # Reason breakdowns (rolling window)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    retry_reasons: dict[str, int] = field(default_factory=dict)
    recent_rejections: list[ExecutionIssue] = field(default_factory=list)
    recent_retries: list[ExecutionIssue] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.submissions_total == 0:
            return 100.0
        return (self.submissions_success / self.submissions_total) * 100

    @property
    def is_healthy(self) -> bool:
        """Check if execution metrics indicate healthy state."""
        return self.success_rate >= 95.0 and self.retry_rate < 0.5

    @property
    def top_rejection_reasons(self) -> list[tuple[str, int]]:
        """Get rejection reasons sorted by count (highest first)."""
        return sorted(self.rejection_reasons.items(), key=lambda x: -x[1])

    @property
    def top_retry_reasons(self) -> list[tuple[str, int]]:
        """Get retry reasons sorted by count (highest first)."""
        return sorted(self.retry_reasons.items(), key=lambda x: -x[1])


@dataclass
class SubmissionRecord:
    """Record of a single order submission."""

    timestamp: float
    latency_ms: float
    success: bool
    rejected: bool = False
    retry_count: int = 0
    failure_reason: str = ""
    rejection_reason: str = ""  # Specific reason for rejection (e.g., rate_limit)
    reason_detail: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    price: float = 0.0


class ExecutionTelemetryCollector:
    """Collects and aggregates execution telemetry.

    Thread-safe collector that maintains rolling windows of submission
    metrics for real-time monitoring.

    Usage:
        collector = get_execution_telemetry()
        collector.record_submission(latency_ms=45.2, success=True)
        metrics = collector.get_metrics()
    """

    def __init__(self, window_size: int = METRICS_WINDOW_SIZE) -> None:
        """Initialize the collector.

        Args:
            window_size: Max submissions to track in rolling window.
        """
        self._window_size = window_size
        self._submissions: deque[SubmissionRecord] = deque(maxlen=window_size)
        self._latencies: deque[float] = deque(maxlen=LATENCY_WINDOW_SIZE)
        self._retry_reasons: deque[str] = deque(maxlen=window_size)
        self._recent_rejections: deque[ExecutionIssue] = deque(maxlen=ISSUES_WINDOW_SIZE)
        self._recent_retries: deque[ExecutionIssue] = deque(maxlen=ISSUES_WINDOW_SIZE)
        self._lock = threading.Lock()

        # Counters for total lifetime stats
        self._total_submissions = 0
        self._total_retries = 0

    def record_submission(
        self,
        latency_ms: float,
        success: bool,
        rejected: bool = False,
        retry_count: int = 0,
        failure_reason: str = "",
        rejection_reason: str = "",
        reason_detail: str = "",
        symbol: str = "",
        side: str = "",
        quantity: float = 0.0,
        price: float = 0.0,
    ) -> None:
        """Record an order submission.

        Args:
            latency_ms: Time from submission to broker response.
            success: Whether the submission succeeded.
            rejected: Whether broker rejected the order.
            retry_count: Number of retries before this result.
            failure_reason: Reason for failure (if failed).
            rejection_reason: Categorized reason for rejection/failure (normalized).
            reason_detail: Additional context for rejection/failure (optional).
            symbol: Order symbol (if available).
            side: Order side (if available).
            quantity: Order quantity (if available).
            price: Order price (if available).
        """
        record = SubmissionRecord(
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=success,
            rejected=rejected,
            retry_count=retry_count,
            failure_reason=failure_reason,
            rejection_reason=rejection_reason,
            reason_detail=reason_detail,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
        )

        with self._lock:
            self._submissions.append(record)
            if success:
                self._latencies.append(latency_ms)
            else:
                issue_reason = rejection_reason or failure_reason or "unknown"
                self._recent_rejections.append(
                    ExecutionIssue(
                        timestamp=record.timestamp,
                        symbol=record.symbol,
                        side=record.side,
                        quantity=record.quantity,
                        price=record.price,
                        reason=issue_reason,
                        reason_detail=record.reason_detail,
                        is_retry=False,
                    )
                )
            self._total_submissions += 1
            self._total_retries += retry_count

    def record_retry(
        self,
        reason: str = "",
        symbol: str = "",
        side: str = "",
        quantity: float = 0.0,
        price: float = 0.0,
    ) -> None:
        """Record a retry attempt (called during retry loop).

        Args:
            reason: Categorized reason for retry (e.g., timeout, connection,
                rate_limit, network, unknown).
            symbol: Order symbol (if available).
            side: Order side (if available).
            quantity: Order quantity (if available).
            price: Order price (if available).
        """
        with self._lock:
            self._total_retries += 1
            if reason:
                self._retry_reasons.append(reason)
                self._recent_retries.append(
                    ExecutionIssue(
                        timestamp=time.time(),
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        reason=reason,
                        reason_detail="",
                        is_retry=True,
                    )
                )

    def get_metrics(self) -> ExecutionMetrics:
        """Get current execution metrics.

        Returns:
            ExecutionMetrics with aggregated statistics.
        """
        with self._lock:
            if not self._submissions:
                return ExecutionMetrics()

            # Count submissions in window
            total = len(self._submissions)
            success = sum(1 for s in self._submissions if s.success)
            failed = sum(1 for s in self._submissions if not s.success and not s.rejected)
            rejected = sum(1 for s in self._submissions if s.rejected)
            retries = sum(s.retry_count for s in self._submissions)

            # Calculate latency percentiles
            latencies = list(self._latencies)
            if latencies:
                avg_latency = statistics.mean(latencies)
                p50_latency = statistics.median(latencies)
                # Calculate p95
                sorted_latencies = sorted(latencies)
                p95_idx = int(len(sorted_latencies) * 0.95)
                p95_latency = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
            else:
                avg_latency = p50_latency = p95_latency = 0.0

            # Get last submission info
            last = self._submissions[-1]
            last_latency = last.latency_ms
            last_time = last.timestamp
            last_reason = ""
            for record in reversed(self._submissions):
                if record.failure_reason:
                    last_reason = record.failure_reason
                    break

            # Calculate retry rate
            retry_rate = retries / total if total > 0 else 0.0

            # Aggregate rejection reasons from submissions in window
            rejection_reasons: dict[str, int] = {}
            for record in self._submissions:
                if record.rejection_reason:
                    rejection_reasons[record.rejection_reason] = (
                        rejection_reasons.get(record.rejection_reason, 0) + 1
                    )

            # Aggregate retry reasons from retry_reasons deque
            retry_reasons: dict[str, int] = {}
            for reason in self._retry_reasons:
                retry_reasons[reason] = retry_reasons.get(reason, 0) + 1

            recent_rejections = list(reversed(self._recent_rejections))
            recent_retries = list(reversed(self._recent_retries))

            return ExecutionMetrics(
                submissions_total=total,
                submissions_success=success,
                submissions_failed=failed,
                submissions_rejected=rejected,
                avg_latency_ms=avg_latency,
                p50_latency_ms=p50_latency,
                p95_latency_ms=p95_latency,
                last_latency_ms=last_latency,
                retry_total=retries,
                retry_rate=retry_rate,
                last_submission_time=last_time,
                last_failure_reason=last_reason,
                rejection_reasons=rejection_reasons,
                retry_reasons=retry_reasons,
                recent_rejections=recent_rejections,
                recent_retries=recent_retries,
            )

    def clear(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._submissions.clear()
            self._latencies.clear()
            self._retry_reasons.clear()
            self._recent_rejections.clear()
            self._recent_retries.clear()
            self._total_submissions = 0
            self._total_retries = 0


# Global singleton
_execution_telemetry: ExecutionTelemetryCollector | None = None


def get_execution_telemetry() -> ExecutionTelemetryCollector:
    """Get or create the global execution telemetry collector."""
    global _execution_telemetry
    if _execution_telemetry is None:
        _execution_telemetry = ExecutionTelemetryCollector()
    return _execution_telemetry


def clear_execution_telemetry() -> None:
    """Clear the global execution telemetry collector (for testing)."""
    global _execution_telemetry
    _execution_telemetry = None


__all__ = [
    "ExecutionIssue",
    "ExecutionMetrics",
    "ExecutionTelemetryCollector",
    "SubmissionRecord",
    "clear_execution_telemetry",
    "get_execution_telemetry",
]
