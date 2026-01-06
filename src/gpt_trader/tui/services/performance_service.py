"""
TUI Performance Monitoring Service.

Provides centralized performance metrics collection and reporting for the TUI.
Bridges existing performance utilities with TUI-specific instrumentation.

Set TUI_PERF_TRACE=1 to enable verbose performance tracing to logs.
"""

from __future__ import annotations

import os
import time
from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.performance import (
    PerformanceCollector,
    PerformanceMonitor,
    ResourceMonitor,
    get_collector,
    get_resource_monitor,
)

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")

# Environment variable to enable verbose performance tracing
# Usage: TUI_PERF_TRACE=1 uv run gpt-trader tui --demo
PERF_TRACE_ENABLED = os.environ.get("TUI_PERF_TRACE", "").lower() in ("1", "true", "yes")


def perf_trace(operation: str, duration_ms: float, **kwargs: Any) -> None:
    """Log a performance trace message if tracing is enabled.

    Args:
        operation: Name of the operation being traced.
        duration_ms: Duration in milliseconds.
        **kwargs: Additional context to include in the trace.
    """
    if not PERF_TRACE_ENABLED:
        return

    extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
    if extra:
        logger.info(f"perf: {operation} {duration_ms:.1f}ms {extra}")
    else:
        logger.info(f"perf: {operation} {duration_ms:.1f}ms")


@dataclass
class FrameMetrics:
    """Metrics for a single frame/update cycle."""

    timestamp: float
    total_duration: float
    state_update_duration: float = 0.0
    widget_render_duration: float = 0.0
    broadcast_duration: float = 0.0


@dataclass
class PerformanceBudget:
    """Performance budget thresholds for TUI metrics.

    Based on TUI_STYLE_GUIDE.md performance requirements.
    """

    # FPS thresholds
    fps_target: float = 0.5
    fps_warning: float = 0.2

    # Frame timing thresholds (seconds)
    avg_frame_time_target: float = 0.050  # 50ms
    avg_frame_time_warning: float = 0.100  # 100ms
    p95_frame_time_target: float = 0.100  # 100ms
    p95_frame_time_warning: float = 0.200  # 200ms
    max_frame_time_target: float = 0.200  # 200ms
    max_frame_time_warning: float = 0.500  # 500ms

    # Memory thresholds (percent)
    memory_percent_target: float = 50.0
    memory_percent_warning: float = 80.0

    # CPU thresholds (percent)
    cpu_percent_target: float = 50.0
    cpu_percent_warning: float = 80.0


# Default performance budgets
DEFAULT_BUDGET = PerformanceBudget()


@dataclass
class BudgetStatus:
    """Status of a metric against its budget."""

    metric_name: str
    value: float
    target: float
    warning: float
    status: str  # "good", "warning", "critical"


@dataclass
class PerformanceSnapshot:
    """Point-in-time performance snapshot for widget display."""

    fps: float = 0.0
    avg_frame_time: float = 0.0
    p95_frame_time: float = 0.0
    max_frame_time: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    throttler_batch_size: float = 0.0
    throttler_queue_depth: int = 0
    slow_operations: list[tuple[str, float]] = field(default_factory=list)
    total_frames: int = 0
    uptime_seconds: float = 0.0

    # Budget evaluation results
    budget_statuses: list[BudgetStatus] = field(default_factory=list)
    budget_violations: int = 0

    def evaluate_budget(self, budget: PerformanceBudget | None = None) -> None:
        """Evaluate metrics against performance budget.

        Args:
            budget: Budget to evaluate against. Uses DEFAULT_BUDGET if None.
        """
        if budget is None:
            budget = DEFAULT_BUDGET

        self.budget_statuses = []
        self.budget_violations = 0

        # FPS (inverted: higher is better)
        fps_status = self._evaluate_inverted("FPS", self.fps, budget.fps_target, budget.fps_warning)
        self.budget_statuses.append(fps_status)
        if fps_status.status == "critical":
            self.budget_violations += 1

        # Average frame time
        avg_status = self._evaluate(
            "Avg Frame Time",
            self.avg_frame_time,
            budget.avg_frame_time_target,
            budget.avg_frame_time_warning,
        )
        self.budget_statuses.append(avg_status)
        if avg_status.status == "critical":
            self.budget_violations += 1

        # P95 frame time
        p95_status = self._evaluate(
            "P95 Frame Time",
            self.p95_frame_time,
            budget.p95_frame_time_target,
            budget.p95_frame_time_warning,
        )
        self.budget_statuses.append(p95_status)
        if p95_status.status == "critical":
            self.budget_violations += 1

        # Max frame time
        max_status = self._evaluate(
            "Max Frame Time",
            self.max_frame_time,
            budget.max_frame_time_target,
            budget.max_frame_time_warning,
        )
        self.budget_statuses.append(max_status)
        if max_status.status == "critical":
            self.budget_violations += 1

        # Memory percent
        mem_status = self._evaluate(
            "Memory %",
            self.memory_percent,
            budget.memory_percent_target,
            budget.memory_percent_warning,
        )
        self.budget_statuses.append(mem_status)
        if mem_status.status == "critical":
            self.budget_violations += 1

        # CPU percent
        cpu_status = self._evaluate(
            "CPU %",
            self.cpu_percent,
            budget.cpu_percent_target,
            budget.cpu_percent_warning,
        )
        self.budget_statuses.append(cpu_status)
        if cpu_status.status == "critical":
            self.budget_violations += 1

    @staticmethod
    def _evaluate(name: str, value: float, target: float, warning: float) -> BudgetStatus:
        """Evaluate a metric (lower is better)."""
        if value <= target:
            status = "good"
        elif value <= warning:
            status = "warning"
        else:
            status = "critical"
        return BudgetStatus(name, value, target, warning, status)

    @staticmethod
    def _evaluate_inverted(name: str, value: float, target: float, warning: float) -> BudgetStatus:
        """Evaluate a metric (higher is better)."""
        if value >= target:
            status = "good"
        elif value >= warning:
            status = "warning"
        else:
            status = "critical"
        return BudgetStatus(name, value, target, warning, status)


class _NoOpContext:
    """No-op context manager when monitoring is disabled."""

    def __enter__(self) -> _NoOpContext:
        return self

    def __exit__(self, *args: object) -> None:
        pass


class TuiPerformanceService:
    """
    Performance monitoring service for the TUI.

    Collects timing metrics from instrumented paths and provides
    aggregated snapshots for the performance dashboard widget.

    Uses existing performance utilities infrastructure:
    - PerformanceCollector for metric storage
    - ResourceMonitor for system metrics (psutil)
    - PerformanceMonitor facade for timing operations
    """

    # Slow operation threshold (seconds)
    SLOW_THRESHOLD = 0.050  # 50ms

    # Maximum frame history for FPS calculation
    MAX_FRAME_HISTORY = 120  # 2 minutes at 1 FPS target

    # Maximum slow operations to track
    MAX_SLOW_OPERATIONS = 20

    def __init__(self, app: TraderApp | None = None, enabled: bool = True) -> None:
        """Initialize the performance service.

        Args:
            app: Optional TraderApp reference for accessing throttler stats.
            enabled: Whether monitoring is enabled. When False, all operations
                    are no-ops with minimal overhead.
        """
        self.app = app
        self.enabled = enabled
        self._start_time = time.time()

        # Use existing infrastructure
        self._collector: PerformanceCollector = get_collector()
        self._resource_monitor: ResourceMonitor = get_resource_monitor()
        self._monitor = PerformanceMonitor(
            collector=self._collector,
            resource_monitor=self._resource_monitor,
        )

        # Frame tracking for FPS
        self._frame_times: deque[float] = deque(maxlen=self.MAX_FRAME_HISTORY)
        self._frame_metrics: deque[FrameMetrics] = deque(maxlen=60)

        # Slow operations log (name, duration, timestamp)
        self._slow_operations: deque[tuple[str, float, float]] = deque(
            maxlen=self.MAX_SLOW_OPERATIONS
        )

        # Total frame counter
        self._total_frames = 0

        logger.debug("TuiPerformanceService initialized (enabled=%s)", enabled)

    def time_operation(self, name: str) -> AbstractContextManager[None] | _NoOpContext:
        """Context manager for timing an operation.

        Args:
            name: Operation name (e.g., "state_update", "widget_render").
                  Will be prefixed with "tui." automatically.

        Returns:
            Context manager that records timing to collector.
            Returns no-op context when monitoring is disabled.
        """
        if not self.enabled:
            return _NoOpContext()
        return self._monitor.time(f"tui.{name}")

    def record_frame(self, metrics: FrameMetrics) -> None:
        """Record a completed frame/update cycle.

        Args:
            metrics: Frame metrics including timing breakdown.
        """
        if not self.enabled:
            return

        self._frame_times.append(metrics.timestamp)
        self._frame_metrics.append(metrics)
        self._total_frames += 1

        # Track slow frames
        if metrics.total_duration > self.SLOW_THRESHOLD:
            self._slow_operations.append(("frame", metrics.total_duration, metrics.timestamp))

    def record_slow_operation(self, name: str, duration: float) -> None:
        """Record a slow operation for dashboard display.

        Args:
            name: Operation name.
            duration: Duration in seconds.
        """
        if not self.enabled:
            return

        if duration > self.SLOW_THRESHOLD:
            self._slow_operations.append((name, duration, time.time()))
            logger.debug("Slow operation detected: %s took %.1fms", name, duration * 1000)

    def get_snapshot(self) -> PerformanceSnapshot:
        """Get current performance snapshot for dashboard display.

        Returns:
            PerformanceSnapshot with current metrics.
        """
        snapshot = PerformanceSnapshot()
        snapshot.total_frames = self._total_frames
        snapshot.uptime_seconds = time.time() - self._start_time

        # Calculate FPS from frame times
        if len(self._frame_times) >= 2:
            time_span = self._frame_times[-1] - self._frame_times[0]
            if time_span > 0:
                snapshot.fps = (len(self._frame_times) - 1) / time_span

        # Calculate frame time statistics
        if self._frame_metrics:
            durations = [m.total_duration for m in self._frame_metrics]
            snapshot.avg_frame_time = sum(durations) / len(durations)
            snapshot.max_frame_time = max(durations)

            # P95 calculation
            sorted_durations = sorted(durations)
            p95_index = int(len(sorted_durations) * 0.95)
            snapshot.p95_frame_time = sorted_durations[min(p95_index, len(sorted_durations) - 1)]

        # System metrics from ResourceMonitor
        if self._resource_monitor.is_available():
            memory = self._resource_monitor.get_memory_usage()
            snapshot.memory_mb = memory.get("rss_mb", 0.0)
            snapshot.memory_percent = memory.get("percent", 0.0)

            cpu = self._resource_monitor.get_cpu_usage()
            snapshot.cpu_percent = cpu.get("cpu_percent", 0.0)

        # Throttler stats (if available via app)
        if self.app is not None:
            ui_coordinator = getattr(self.app, "ui_coordinator", None)
            if ui_coordinator is not None:
                throttler = getattr(ui_coordinator, "_throttler", None)
                if throttler is not None:
                    stats = throttler.get_stats()
                    snapshot.throttler_batch_size = stats.average_batch_size
                    pending_count = getattr(throttler, "pending_count", 0)
                    snapshot.throttler_queue_depth = pending_count

        # Recent slow operations (most recent 5)
        snapshot.slow_operations = [
            (name, duration) for name, duration, _ in list(self._slow_operations)[-5:]
        ]

        # Evaluate against performance budget
        snapshot.evaluate_budget()

        return snapshot

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get full summary from underlying collector.

        Returns:
            Dictionary of operation names to their statistics.
        """
        return self._monitor.summary()

    def get_operation_stats(self, name: str) -> dict[str, float]:
        """Get stats for a specific operation.

        Args:
            name: Operation name (without "tui." prefix).

        Returns:
            Dictionary with count, avg, min, max, total.
        """
        summary = self._monitor.summary()
        return summary.get(f"tui.{name}", {})

    def reset(self) -> None:
        """Reset all metrics."""
        self._frame_times.clear()
        self._frame_metrics.clear()
        self._slow_operations.clear()
        self._total_frames = 0
        self._start_time = time.time()
        self._collector.clear()
        logger.debug("TuiPerformanceService metrics reset")

    def system_snapshot(self) -> dict[str, Any]:
        """Get detailed system resource information.

        Returns:
            Dictionary with system info if psutil is available.
        """
        return self._monitor.system_snapshot()


# Global singleton
_performance_service: TuiPerformanceService | None = None


def get_tui_performance_service() -> TuiPerformanceService:
    """Get or create the global TUI performance service.

    Returns:
        The global TuiPerformanceService instance.
    """
    global _performance_service
    if _performance_service is None:
        _performance_service = TuiPerformanceService()
    return _performance_service


def set_tui_performance_service(service: TuiPerformanceService) -> None:
    """Set the global TUI performance service.

    Used by TraderApp during initialization to set up the service
    with proper app reference.

    Args:
        service: The TuiPerformanceService instance to use globally.
    """
    global _performance_service
    # Clean up old service before replacing
    if _performance_service is not None and _performance_service is not service:
        _performance_service.reset()
        _performance_service.app = None  # Clear app reference
    _performance_service = service


def clear_tui_performance_service() -> None:
    """Clear the global TUI performance service.

    Called during app shutdown to release resources and prevent
    memory accumulation across multiple TUI runs.
    """
    global _performance_service
    if _performance_service is not None:
        _performance_service.reset()
        _performance_service.app = None
        _performance_service = None
        logger.debug("TUI performance service cleared")


__all__ = [
    "BudgetStatus",
    "DEFAULT_BUDGET",
    "FrameMetrics",
    "PERF_TRACE_ENABLED",
    "PerformanceBudget",
    "PerformanceSnapshot",
    "TuiPerformanceService",
    "clear_tui_performance_service",
    "get_tui_performance_service",
    "perf_trace",
    "set_tui_performance_service",
]
