"""Tests for TUI performance monitoring service."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.tui.services.performance_service import (
    FrameMetrics,
    TuiPerformanceService,
    _NoOpContext,
    get_tui_performance_service,
    set_tui_performance_service,
)


def _patch_time(monkeypatch, values: list[float]) -> None:
    import gpt_trader.utilities.performance.timing as timing_module

    iterator = iter(values)
    monkeypatch.setattr(timing_module.time, "time", lambda: next(iterator))


class TestTuiPerformanceService:
    """Tests for TuiPerformanceService."""

    @pytest.fixture
    def service(self) -> TuiPerformanceService:
        return TuiPerformanceService(app=None, enabled=True)

    @pytest.fixture
    def disabled_service(self) -> TuiPerformanceService:
        return TuiPerformanceService(app=None, enabled=False)

    def test_init_enabled(self, service: TuiPerformanceService) -> None:
        assert service.enabled is True
        assert service.app is None

    def test_init_disabled(self, disabled_service: TuiPerformanceService) -> None:
        assert disabled_service.enabled is False

    def test_time_operation_enabled(self, service: TuiPerformanceService, monkeypatch) -> None:
        _patch_time(monkeypatch, [1000.0, 1000.02])
        with service.time_operation("test_op"):
            pass

        summary = service.get_summary()
        assert "tui.test_op" in summary
        assert summary["tui.test_op"]["count"] == 1
        assert summary["tui.test_op"]["avg"] >= 0.01

    def test_time_operation_disabled(self, disabled_service: TuiPerformanceService) -> None:
        context = disabled_service.time_operation("test_op")
        assert isinstance(context, _NoOpContext)

    def test_disabled_service_minimal_overhead(
        self, disabled_service: TuiPerformanceService
    ) -> None:
        start = time.time()
        for _ in range(1000):
            with disabled_service.time_operation("noop"):
                pass
        duration = time.time() - start

        assert duration < 0.050

    def test_record_frame(self, service: TuiPerformanceService) -> None:
        metrics = FrameMetrics(
            timestamp=time.time(),
            total_duration=0.050,
            state_update_duration=0.020,
            widget_render_duration=0.030,
        )

        service.record_frame(metrics)

        assert service._total_frames == 1
        assert len(service._frame_metrics) == 1

    def test_record_frame_disabled(self, disabled_service: TuiPerformanceService) -> None:
        metrics = FrameMetrics(timestamp=time.time(), total_duration=0.050)

        disabled_service.record_frame(metrics)

        assert disabled_service._total_frames == 0

    def test_fps_calculation(self, service: TuiPerformanceService) -> None:
        base_time = time.time()

        for i in range(10):
            metrics = FrameMetrics(
                timestamp=base_time + i,
                total_duration=0.050,
            )
            service.record_frame(metrics)

        snapshot = service.get_snapshot()

        assert 0.9 <= snapshot.fps <= 1.1

    def test_frame_time_statistics(self, service: TuiPerformanceService) -> None:
        base_time = time.time()

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
        service.record_slow_operation("slow_render", 0.100)

        snapshot = service.get_snapshot()

        assert len(snapshot.slow_operations) == 1
        assert snapshot.slow_operations[0][0] == "slow_render"
        assert snapshot.slow_operations[0][1] == 0.100

    def test_slow_operation_not_tracked_below_threshold(
        self, service: TuiPerformanceService
    ) -> None:
        service.record_slow_operation("fast_render", 0.010)

        snapshot = service.get_snapshot()

        assert len(snapshot.slow_operations) == 0

    def test_slow_operation_disabled(self, disabled_service: TuiPerformanceService) -> None:
        disabled_service.record_slow_operation("slow_render", 0.100)

        assert len(disabled_service._slow_operations) == 0

    def test_get_operation_stats(self, service: TuiPerformanceService, monkeypatch) -> None:
        _patch_time(monkeypatch, [1000.0, 1000.02])
        with service.time_operation("my_operation"):
            pass

        stats = service.get_operation_stats("my_operation")

        assert stats["count"] == 1
        assert stats["avg"] >= 0.01

    def test_reset(self, service: TuiPerformanceService) -> None:
        service.record_frame(FrameMetrics(timestamp=time.time(), total_duration=0.050))
        service.record_slow_operation("slow", 0.100)

        assert service._total_frames == 1
        assert len(service._slow_operations) == 1

        service.reset()

        assert service._total_frames == 0
        assert len(service._frame_metrics) == 0
        assert len(service._slow_operations) == 0

    def test_snapshot_with_app_reference(self) -> None:
        mock_app = MagicMock()
        mock_throttler = MagicMock()
        mock_throttler.get_stats.return_value = MagicMock(average_batch_size=2.5)
        mock_throttler.pending_count = 3
        mock_app.ui_coordinator._throttler = mock_throttler

        service = TuiPerformanceService(app=mock_app, enabled=True)
        snapshot = service.get_snapshot()

        assert snapshot.throttler_batch_size == 2.5
        assert snapshot.throttler_queue_depth == 3

    @patch("gpt_trader.tui.services.performance_service.get_resource_monitor")
    def test_snapshot_memory_metrics(self, mock_get_monitor) -> None:
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
        import gpt_trader.tui.services.performance_service as module

        module._performance_service = None

        service = get_tui_performance_service()

        assert service is not None
        assert isinstance(service, TuiPerformanceService)

    def test_set_and_get_service(self) -> None:
        custom_service = TuiPerformanceService(app=None, enabled=False)

        set_tui_performance_service(custom_service)
        retrieved = get_tui_performance_service()

        assert retrieved is custom_service
        assert retrieved.enabled is False
