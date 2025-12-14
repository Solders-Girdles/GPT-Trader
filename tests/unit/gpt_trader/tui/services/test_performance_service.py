"""Tests for TUI performance monitoring service."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.tui.services.performance_service import (
    FrameMetrics,
    PerformanceSnapshot,
    TuiPerformanceService,
    _NoOpContext,
    get_tui_performance_service,
    set_tui_performance_service,
)


class TestFrameMetrics:
    """Tests for FrameMetrics dataclass."""

    def test_create_frame_metrics(self) -> None:
        """Test creating a FrameMetrics instance."""
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
        """Test FrameMetrics default values."""
        metrics = FrameMetrics(timestamp=1000.0, total_duration=0.050)

        assert metrics.state_update_duration == 0.0
        assert metrics.widget_render_duration == 0.0
        assert metrics.broadcast_duration == 0.0


class TestPerformanceSnapshot:
    """Tests for PerformanceSnapshot dataclass."""

    def test_create_snapshot(self) -> None:
        """Test creating a PerformanceSnapshot instance."""
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
        """Test PerformanceSnapshot default values."""
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
        """Test that _NoOpContext works as a context manager."""
        context = _NoOpContext()

        with context as result:
            assert result is context

    def test_noop_context_multiple_uses(self) -> None:
        """Test that _NoOpContext can be reused."""
        context = _NoOpContext()

        for _ in range(3):
            with context:
                pass


class TestTuiPerformanceService:
    """Tests for TuiPerformanceService."""

    @pytest.fixture
    def service(self) -> TuiPerformanceService:
        """Create a fresh performance service for each test."""
        return TuiPerformanceService(app=None, enabled=True)

    @pytest.fixture
    def disabled_service(self) -> TuiPerformanceService:
        """Create a disabled performance service."""
        return TuiPerformanceService(app=None, enabled=False)

    def test_init_enabled(self, service: TuiPerformanceService) -> None:
        """Test initializing an enabled service."""
        assert service.enabled is True
        assert service.app is None

    def test_init_disabled(self, disabled_service: TuiPerformanceService) -> None:
        """Test initializing a disabled service."""
        assert disabled_service.enabled is False

    def test_time_operation_enabled(self, service: TuiPerformanceService) -> None:
        """Test timing context manager when enabled."""
        with service.time_operation("test_op"):
            time.sleep(0.01)  # 10ms

        summary = service.get_summary()
        assert "tui.test_op" in summary
        assert summary["tui.test_op"]["count"] == 1
        assert summary["tui.test_op"]["avg"] >= 0.01

    def test_time_operation_disabled(self, disabled_service: TuiPerformanceService) -> None:
        """Test timing context manager returns no-op when disabled."""
        context = disabled_service.time_operation("test_op")
        assert isinstance(context, _NoOpContext)

    def test_disabled_service_minimal_overhead(
        self, disabled_service: TuiPerformanceService
    ) -> None:
        """Test disabled service has minimal overhead."""
        start = time.time()
        for _ in range(1000):
            with disabled_service.time_operation("noop"):
                pass
        duration = time.time() - start

        # Should complete in under 50ms for 1000 iterations
        # (being generous to avoid flaky tests)
        assert duration < 0.050

    def test_record_frame(self, service: TuiPerformanceService) -> None:
        """Test recording frame metrics."""
        metrics = FrameMetrics(
            timestamp=time.time(),
            total_duration=0.050,
            state_update_duration=0.020,
            widget_render_duration=0.030,
        )

        service.record_frame(metrics)

        assert service._total_frames == 1
        assert len(service._frame_metrics) == 1

    def test_record_frame_disabled(
        self, disabled_service: TuiPerformanceService
    ) -> None:
        """Test recording frame metrics when disabled does nothing."""
        metrics = FrameMetrics(timestamp=time.time(), total_duration=0.050)

        disabled_service.record_frame(metrics)

        assert disabled_service._total_frames == 0

    def test_fps_calculation(self, service: TuiPerformanceService) -> None:
        """Test FPS is calculated from frame times."""
        base_time = time.time()

        # Record 10 frames over ~10 seconds (1 FPS)
        for i in range(10):
            metrics = FrameMetrics(
                timestamp=base_time + i,
                total_duration=0.050,
            )
            service.record_frame(metrics)

        snapshot = service.get_snapshot()

        # FPS should be approximately 1.0 (9 intervals over 9 seconds)
        assert 0.9 <= snapshot.fps <= 1.1

    def test_frame_time_statistics(self, service: TuiPerformanceService) -> None:
        """Test frame time statistics calculation."""
        base_time = time.time()

        # Record frames with varying durations
        durations = [0.010, 0.020, 0.030, 0.040, 0.050]
        for i, duration in enumerate(durations):
            metrics = FrameMetrics(
                timestamp=base_time + i,
                total_duration=duration,
            )
            service.record_frame(metrics)

        snapshot = service.get_snapshot()

        assert snapshot.avg_frame_time == pytest.approx(0.030, rel=0.01)
        assert snapshot.max_frame_time == 0.050

    def test_slow_operation_tracking(self, service: TuiPerformanceService) -> None:
        """Test slow operations are tracked."""
        # Record a slow operation (above 50ms threshold)
        service.record_slow_operation("slow_render", 0.100)

        snapshot = service.get_snapshot()

        assert len(snapshot.slow_operations) == 1
        assert snapshot.slow_operations[0][0] == "slow_render"
        assert snapshot.slow_operations[0][1] == 0.100

    def test_slow_operation_not_tracked_below_threshold(
        self, service: TuiPerformanceService
    ) -> None:
        """Test operations below threshold are not tracked."""
        # Record a fast operation (below 50ms threshold)
        service.record_slow_operation("fast_render", 0.010)

        snapshot = service.get_snapshot()

        assert len(snapshot.slow_operations) == 0

    def test_slow_operation_disabled(
        self, disabled_service: TuiPerformanceService
    ) -> None:
        """Test slow operations not recorded when disabled."""
        disabled_service.record_slow_operation("slow_render", 0.100)

        # Still 0 because disabled
        assert len(disabled_service._slow_operations) == 0

    def test_get_operation_stats(self, service: TuiPerformanceService) -> None:
        """Test getting stats for a specific operation."""
        with service.time_operation("my_operation"):
            time.sleep(0.01)

        stats = service.get_operation_stats("my_operation")

        assert stats["count"] == 1
        assert stats["avg"] >= 0.01

    def test_reset(self, service: TuiPerformanceService) -> None:
        """Test resetting all metrics."""
        # Record some data
        service.record_frame(
            FrameMetrics(timestamp=time.time(), total_duration=0.050)
        )
        service.record_slow_operation("slow", 0.100)

        assert service._total_frames == 1
        assert len(service._slow_operations) == 1

        service.reset()

        assert service._total_frames == 0
        assert len(service._frame_metrics) == 0
        assert len(service._slow_operations) == 0

    def test_snapshot_with_app_reference(self) -> None:
        """Test snapshot includes throttler stats when app is available."""
        mock_app = MagicMock()
        mock_throttler = MagicMock()
        mock_throttler.get_stats.return_value = MagicMock(average_batch_size=2.5)
        mock_throttler.pending_count = 3
        mock_app.ui_coordinator._throttler = mock_throttler

        service = TuiPerformanceService(app=mock_app, enabled=True)
        snapshot = service.get_snapshot()

        assert snapshot.throttler_batch_size == 2.5
        assert snapshot.throttler_queue_depth == 3

    @patch(
        "gpt_trader.tui.services.performance_service.get_resource_monitor"
    )
    def test_snapshot_memory_metrics(self, mock_get_monitor) -> None:
        """Test memory metrics are included in snapshot."""
        mock_monitor = MagicMock()
        mock_monitor.is_available.return_value = True
        mock_monitor.get_memory_usage.return_value = {
            "rss_mb": 256.0,
            "percent": 25.0,
        }
        mock_monitor.get_cpu_usage.return_value = {"cpu_percent": 15.0}
        mock_get_monitor.return_value = mock_monitor

        service = TuiPerformanceService(app=None, enabled=True)
        snapshot = service.get_snapshot()

        assert snapshot.memory_mb == 256.0
        assert snapshot.memory_percent == 25.0
        assert snapshot.cpu_percent == 15.0


class TestGlobalService:
    """Tests for global service functions."""

    def test_get_creates_service(self) -> None:
        """Test get_tui_performance_service creates a service if none exists."""
        # Reset global
        import gpt_trader.tui.services.performance_service as module

        module._performance_service = None

        service = get_tui_performance_service()

        assert service is not None
        assert isinstance(service, TuiPerformanceService)

    def test_set_and_get_service(self) -> None:
        """Test setting and getting the global service."""
        custom_service = TuiPerformanceService(app=None, enabled=False)

        set_tui_performance_service(custom_service)
        retrieved = get_tui_performance_service()

        assert retrieved is custom_service
        assert retrieved.enabled is False
