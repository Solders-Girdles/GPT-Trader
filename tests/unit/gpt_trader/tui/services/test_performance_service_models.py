"""Tests for TUI performance monitoring models."""

from gpt_trader.tui.services.performance_service import (
    FrameMetrics,
    PerformanceSnapshot,
    _NoOpContext,
)


class TestFrameMetrics:
    """Tests for FrameMetrics dataclass."""

    def test_create_frame_metrics(self) -> None:
        metrics = FrameMetrics(
            timestamp=1000.0,
            total_duration=0.050,
            state_update_duration=0.020,
            widget_render_duration=0.030,
        )

        assert metrics.timestamp == 1000.0
        assert metrics.total_duration == 0.050
        assert metrics.state_update_duration == 0.020
        assert metrics.widget_render_duration == 0.030

    def test_frame_metrics_defaults(self) -> None:
        metrics = FrameMetrics(timestamp=1000.0, total_duration=0.050)

        assert metrics.state_update_duration == 0.0
        assert metrics.widget_render_duration == 0.0
        assert metrics.broadcast_duration == 0.0


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        snapshot = PerformanceSnapshot(
            fps=1.0,
            avg_frame_time=0.050,
            p95_frame_time=0.080,
            memory_mb=256.0,
            memory_percent=25.0,
        )

        assert snapshot.fps == 1.0
        assert snapshot.avg_frame_time == 0.050
        assert snapshot.p95_frame_time == 0.080
        assert snapshot.memory_mb == 256.0
        assert snapshot.memory_percent == 25.0

    def test_snapshot_defaults(self) -> None:
        snapshot = PerformanceSnapshot()

        assert snapshot.fps == 0.0
        assert snapshot.avg_frame_time == 0.0
        assert snapshot.p95_frame_time == 0.0
        assert snapshot.max_frame_time == 0.0
        assert snapshot.memory_mb == 0.0
        assert snapshot.memory_percent == 0.0
        assert snapshot.cpu_percent == 0.0
        assert snapshot.throttler_batch_size == 0.0
        assert snapshot.throttler_queue_depth == 0
        assert snapshot.slow_operations == []
        assert snapshot.total_frames == 0
        assert snapshot.uptime_seconds == 0.0


class TestNoOpContext:
    """Tests for the no-op context manager."""

    def test_noop_context_manager(self) -> None:
        context = _NoOpContext()

        with context as result:
            assert result is context

    def test_noop_context_multiple_uses(self) -> None:
        context = _NoOpContext()

        for _ in range(3):
            with context as entered:
                assert entered is context
