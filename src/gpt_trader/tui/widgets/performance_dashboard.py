"""
Performance Dashboard Widget.

Displays real-time TUI performance metrics including FPS, latency,
memory usage, and throttler efficiency. Shows budget status indicators
for each metric based on defined performance targets.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Label, Static

from gpt_trader.tui.services.performance_service import (
    DEFAULT_BUDGET,
    PerformanceSnapshot,
    get_tui_performance_service,
)


class PerformanceDashboardWidget(Static):
    """
    Displays real-time TUI performance metrics with budget status.

    Shows:
    - FPS / refresh rate with budget indicator
    - Update latency (avg, p95, max) with budget status
    - Memory usage with trend and budget
    - Throttler efficiency (batch size, queue depth)
    - Budget violations summary
    - Recent slow operations
    """

    # Refresh interval in seconds
    REFRESH_INTERVAL = 1.0

    # Styles moved to styles/widgets/performance.tcss

    # Reactive property for snapshot data
    snapshot: reactive[PerformanceSnapshot | None] = reactive(None)

    def __init__(
        self,
        compact: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the performance dashboard.

        Args:
            compact: If True, show a condensed view with fewer metrics.
        """
        super().__init__(*args, **kwargs)
        self._compact = compact
        self._refresh_timer: Timer | None = None
        # Cache label references to avoid repeated query_one calls
        self._fps_label: Label | None = None
        self._latency_label: Label | None = None
        self._memory_label: Label | None = None
        self._throttler_label: Label | None = None
        self._slow_ops_container: Vertical | None = None

    def compose(self) -> ComposeResult:
        yield Label("PERFORMANCE", classes="perf-header")

        # Budget status row (shows overall health)
        with Horizontal(classes="perf-row budget-summary"):
            yield Label("Budget:", classes="perf-label")
            yield Label("--", id="perf-budget-status", classes="perf-value")

        # FPS row with target
        with Horizontal(classes="perf-row"):
            yield Label("FPS:", classes="perf-label")
            yield Label("--", id="perf-fps", classes="perf-value")
            yield Label(f"(≥{DEFAULT_BUDGET.fps_target})", classes="perf-target")

        # Latency row with target
        with Horizontal(classes="perf-row"):
            yield Label("Latency:", classes="perf-label")
            yield Label("--", id="perf-latency", classes="perf-value")
            yield Label(f"(<{int(DEFAULT_BUDGET.avg_frame_time_target * 1000)}ms)", classes="perf-target")

        # Memory row with target
        with Horizontal(classes="perf-row"):
            yield Label("Memory:", classes="perf-label")
            yield Label("--", id="perf-memory", classes="perf-value")
            yield Label(f"(<{int(DEFAULT_BUDGET.memory_percent_target)}%)", classes="perf-target")

        if not self._compact:
            # CPU row (expanded mode only)
            with Horizontal(classes="perf-row"):
                yield Label("CPU:", classes="perf-label")
                yield Label("--", id="perf-cpu", classes="perf-value")
                yield Label(f"(<{int(DEFAULT_BUDGET.cpu_percent_target)}%)", classes="perf-target")

            # Throttler row (expanded mode only)
            with Horizontal(classes="perf-row"):
                yield Label("Throttler:", classes="perf-label")
                yield Label("--", id="perf-throttler", classes="perf-value")

            # Frames row
            with Horizontal(classes="perf-row"):
                yield Label("Frames:", classes="perf-label")
                yield Label("--", id="perf-frames", classes="perf-value")

            # Slow operations section
            yield Label("Slow Operations:", classes="slow-ops-header")
            yield Vertical(id="slow-ops-list")

    def on_mount(self) -> None:
        """Start refresh timer when widget is mounted."""
        # Cache label references for performance
        self._fps_label = self.query_one("#perf-fps", Label)
        self._latency_label = self.query_one("#perf-latency", Label)
        self._memory_label = self.query_one("#perf-memory", Label)
        self._budget_label: Label | None = None
        try:
            self._budget_label = self.query_one("#perf-budget-status", Label)
        except Exception:
            pass

        if not self._compact:
            try:
                self._throttler_label = self.query_one("#perf-throttler", Label)
                self._slow_ops_container = self.query_one("#slow-ops-list", Vertical)
            except Exception:
                pass

        # Start refresh timer
        self._refresh_timer = self.set_interval(
            self.REFRESH_INTERVAL,
            self._refresh_metrics,
        )

    def on_unmount(self) -> None:
        """Stop refresh timer when widget is unmounted."""
        if self._refresh_timer is not None:
            self._refresh_timer.stop()
            self._refresh_timer = None

    def _refresh_metrics(self) -> None:
        """Fetch and display latest metrics."""
        service = get_tui_performance_service()
        self.snapshot = service.get_snapshot()

    def watch_snapshot(self, snapshot: PerformanceSnapshot | None) -> None:
        """Update display when snapshot changes."""
        if snapshot is None:
            return

        # Update budget status summary
        if self._budget_label is not None:
            violations = snapshot.budget_violations
            if violations == 0:
                self._budget_label.update("✓ All metrics OK")
                self._budget_label.set_classes("perf-value good")
            elif violations <= 2:
                self._budget_label.update(f"⚠ {violations} warning(s)")
                self._budget_label.set_classes("perf-value warning")
            else:
                self._budget_label.update(f"✗ {violations} violation(s)")
                self._budget_label.set_classes("perf-value bad")

        # Update FPS (use cached reference)
        if self._fps_label is not None:
            fps_class = self._get_fps_class(snapshot.fps)
            self._fps_label.update(f"{snapshot.fps:.1f}")
            self._fps_label.set_classes(f"perf-value {fps_class}")

        # Update latency
        if self._latency_label is not None:
            avg_ms = snapshot.avg_frame_time * 1000
            p95_ms = snapshot.p95_frame_time * 1000
            latency_class = self._get_latency_class(avg_ms)
            self._latency_label.update(f"{avg_ms:.0f}ms avg, {p95_ms:.0f}ms p95")
            self._latency_label.set_classes(f"perf-value {latency_class}")

        # Update memory
        if self._memory_label is not None:
            memory_class = self._get_memory_class(snapshot.memory_percent)
            self._memory_label.update(
                f"{snapshot.memory_mb:.0f}MB ({snapshot.memory_percent:.0f}%)"
            )
            self._memory_label.set_classes(f"perf-value {memory_class}")

        if not self._compact:
            self._update_expanded_metrics(snapshot)

    def _update_expanded_metrics(self, snapshot: PerformanceSnapshot) -> None:
        """Update expanded mode-only metrics."""
        # Update CPU
        try:
            cpu_label = self.query_one("#perf-cpu", Label)
            cpu_class = self._get_cpu_class(snapshot.cpu_percent)
            cpu_label.update(f"{snapshot.cpu_percent:.1f}%")
            cpu_label.set_classes(f"perf-value {cpu_class}")
        except Exception:
            pass

        # Update throttler
        if self._throttler_label is not None:
            self._throttler_label.update(
                f"batch={snapshot.throttler_batch_size:.1f}, "
                f"queue={snapshot.throttler_queue_depth}"
            )

        # Update frames
        try:
            frames_label = self.query_one("#perf-frames", Label)
            uptime_min = snapshot.uptime_seconds / 60
            frames_label.update(
                f"{snapshot.total_frames} ({uptime_min:.1f}min)"
            )
        except Exception:
            pass

        # Update slow operations
        if self._slow_ops_container is not None:
            self._update_slow_operations(snapshot)

    def _update_slow_operations(self, snapshot: PerformanceSnapshot) -> None:
        """Update slow operations list."""
        if self._slow_ops_container is None:
            return

        # Clear existing entries
        self._slow_ops_container.remove_children()

        if snapshot.slow_operations:
            # Show most recent 3 slow operations
            for name, duration in snapshot.slow_operations[-3:]:
                self._slow_ops_container.mount(
                    Label(f"{name}: {duration * 1000:.0f}ms", classes="slow-op")
                )
        else:
            self._slow_ops_container.mount(
                Label("None", classes="text-muted")
            )

    @staticmethod
    def _get_fps_class(fps: float) -> str:
        """Get CSS class for FPS value."""
        if fps >= 0.5:
            return "good"
        elif fps >= 0.2:
            return "warning"
        return "bad"

    @staticmethod
    def _get_latency_class(avg_ms: float) -> str:
        """Get CSS class for latency value."""
        if avg_ms < 50:
            return "good"
        elif avg_ms < 100:
            return "warning"
        return "bad"

    @staticmethod
    def _get_memory_class(memory_percent: float) -> str:
        """Get CSS class for memory value."""
        if memory_percent < 50:
            return "good"
        elif memory_percent < 80:
            return "warning"
        return "bad"

    @staticmethod
    def _get_cpu_class(cpu_percent: float) -> str:
        """Get CSS class for CPU value."""
        if cpu_percent < 50:
            return "good"
        elif cpu_percent < 80:
            return "warning"
        return "bad"
