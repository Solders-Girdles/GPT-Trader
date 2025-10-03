"""Performance tracking for logger operations."""

import threading


class PerformanceTracker:
    """Tracks logger performance metrics."""

    def __init__(self):
        """Initialize performance tracker."""
        self._log_count = 0
        self._total_log_time = 0.0
        self._lock = threading.Lock()

    def record(self, duration_seconds: float) -> None:
        """
        Record a log operation duration.

        Args:
            duration_seconds: Operation duration in seconds
        """
        with self._lock:
            self._log_count += 1
            self._total_log_time += duration_seconds

    def get_stats(self) -> dict[str, float | int]:
        """
        Get performance statistics.

        Returns:
            Dictionary with avg_log_time_ms, total_logs, total_log_time_ms
        """
        with self._lock:
            if self._log_count == 0:
                return {"avg_log_time_ms": 0.0, "total_logs": 0}

            avg_time_ms = (self._total_log_time / self._log_count) * 1000
            return {
                "avg_log_time_ms": avg_time_ms,
                "total_logs": self._log_count,
                "total_log_time_ms": self._total_log_time * 1000,
            }
