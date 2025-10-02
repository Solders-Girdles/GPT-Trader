"""
Performance instrumentation for State Management system.

Provides lightweight timing and metrics collection for StateManager
and repository operations to measure and optimize batch performance.
"""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation type."""

    operation_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    min_time_ms: float = float("inf")
    max_time_ms: float = 0.0
    errors: int = 0

    def add_timing(self, duration_ms: float) -> None:
        """Record a successful operation timing."""
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)

    def record_error(self) -> None:
        """Record an operation error."""
        self.errors += 1

    @property
    def avg_time_ms(self) -> float:
        """Calculate average operation time."""
        if self.call_count == 0:
            return 0.0
        return self.total_time_ms / self.call_count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "operation": self.operation_name,
            "calls": self.call_count,
            "total_ms": round(self.total_time_ms, 2),
            "avg_ms": round(self.avg_time_ms, 2),
            "min_ms": round(self.min_time_ms, 2) if self.min_time_ms != float("inf") else 0.0,
            "max_ms": round(self.max_time_ms, 2),
            "errors": self.errors,
        }


@dataclass
class StatePerformanceMetrics:
    """
    Collects and aggregates performance metrics for state operations.

    Tracks timing data for StateManager and repository operations to
    measure performance and validate optimization improvements.
    """

    enabled: bool = False
    _metrics: dict[str, OperationMetrics] = field(default_factory=dict)
    _start_time: datetime = field(default_factory=datetime.utcnow)

    def is_enabled(self) -> bool:
        """Check if metrics collection is enabled."""
        return self.enabled

    @contextmanager
    def time_operation(self, operation_name: str) -> Generator[None, None, None]:
        """
        Context manager to time an operation.

        Usage:
            with metrics.time_operation("get_state"):
                result = await get_state(key)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        error_occurred = False

        try:
            yield
        except Exception:
            error_occurred = True
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000

            if operation_name not in self._metrics:
                self._metrics[operation_name] = OperationMetrics(operation_name)

            if error_occurred:
                self._metrics[operation_name].record_error()
            else:
                self._metrics[operation_name].add_timing(duration_ms)

    def get_metrics(
        self, operation_name: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        Get metrics for specific operation or all operations.

        Args:
            operation_name: Specific operation to query, or None for all

        Returns:
            Single operation dict or list of all operation dicts
        """
        if operation_name:
            if operation_name in self._metrics:
                return self._metrics[operation_name].to_dict()
            return {
                "operation": operation_name,
                "calls": 0,
                "total_ms": 0.0,
                "avg_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "errors": 0,
            }

        return [m.to_dict() for m in self._metrics.values()]

    def get_summary(self) -> dict[str, Any]:
        """Get summary of all collected metrics."""
        total_calls = sum(m.call_count for m in self._metrics.values())
        total_time_ms = sum(m.total_time_ms for m in self._metrics.values())
        total_errors = sum(m.errors for m in self._metrics.values())

        uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()

        return {
            "enabled": self.enabled,
            "uptime_seconds": round(uptime_seconds, 2),
            "total_operations": total_calls,
            "total_time_ms": round(total_time_ms, 2),
            "total_errors": total_errors,
            "operations_per_second": (
                round(total_calls / uptime_seconds, 2) if uptime_seconds > 0 else 0.0
            ),
            "operation_breakdown": self.get_metrics(),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._metrics.clear()
        self._start_time = datetime.utcnow()

    def log_summary(self, level: int = logging.INFO) -> None:
        """Log performance summary."""
        if not self.enabled:
            logger.debug("Performance metrics disabled")
            return

        summary = self.get_summary()
        logger.log(
            level,
            f"State Performance Summary: {summary['total_operations']} ops "
            f"in {summary['uptime_seconds']:.2f}s "
            f"({summary['operations_per_second']:.2f} ops/sec)",
        )

        for op_metrics in summary["operation_breakdown"]:
            logger.log(
                level,
                f"  {op_metrics['operation']}: {op_metrics['calls']} calls, "
                f"avg {op_metrics['avg_ms']:.2f}ms "
                f"(min {op_metrics['min_ms']:.2f}ms, max {op_metrics['max_ms']:.2f}ms)",
            )


# Global instance for convenience
_global_metrics: StatePerformanceMetrics | None = None


def get_global_metrics() -> StatePerformanceMetrics:
    """Get or create global performance metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = StatePerformanceMetrics(enabled=False)
    return _global_metrics


def enable_performance_tracking() -> None:
    """Enable global performance tracking."""
    get_global_metrics().enabled = True
    logger.info("State performance tracking enabled")


def disable_performance_tracking() -> None:
    """Disable global performance tracking."""
    get_global_metrics().enabled = False
    logger.info("State performance tracking disabled")
