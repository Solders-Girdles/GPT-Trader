"""Tests for TUI performance dashboard widget."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.tui.services.performance_service import PerformanceSnapshot
from gpt_trader.tui.widgets.performance_dashboard import PerformanceDashboardWidget


class TestPerformanceDashboardWidget:
    """Tests for the PerformanceDashboardWidget class."""

    def test_init_default_mode(self) -> None:
        """Test widget initializes in default (expanded) mode."""
        widget = PerformanceDashboardWidget()

        assert widget._compact is False
        assert widget._refresh_timer is None

    def test_init_compact_mode(self) -> None:
        """Test widget initializes in compact mode."""
        widget = PerformanceDashboardWidget(compact=True)

        assert widget._compact is True

    def test_fps_class_good(self) -> None:
        """Test FPS >= 0.5 returns 'status-ok' class."""
        assert PerformanceDashboardWidget._get_fps_class(0.5) == "status-ok"
        assert PerformanceDashboardWidget._get_fps_class(1.0) == "status-ok"
        assert PerformanceDashboardWidget._get_fps_class(10.0) == "status-ok"

    def test_fps_class_warning(self) -> None:
        """Test FPS between 0.2 and 0.5 returns 'status-warning' class."""
        assert PerformanceDashboardWidget._get_fps_class(0.2) == "status-warning"
        assert PerformanceDashboardWidget._get_fps_class(0.3) == "status-warning"
        assert PerformanceDashboardWidget._get_fps_class(0.49) == "status-warning"

    def test_fps_class_bad(self) -> None:
        """Test FPS < 0.2 returns 'status-critical' class."""
        assert PerformanceDashboardWidget._get_fps_class(0.0) == "status-critical"
        assert PerformanceDashboardWidget._get_fps_class(0.1) == "status-critical"
        assert PerformanceDashboardWidget._get_fps_class(0.19) == "status-critical"

    def test_latency_class_good(self) -> None:
        """Test latency < 50ms returns 'status-ok' class."""
        assert PerformanceDashboardWidget._get_latency_class(0) == "status-ok"
        assert PerformanceDashboardWidget._get_latency_class(25) == "status-ok"
        assert PerformanceDashboardWidget._get_latency_class(49) == "status-ok"

    def test_latency_class_warning(self) -> None:
        """Test latency between 50-150ms returns 'status-warning' class."""
        assert PerformanceDashboardWidget._get_latency_class(50) == "status-warning"
        assert PerformanceDashboardWidget._get_latency_class(100) == "status-warning"
        assert PerformanceDashboardWidget._get_latency_class(149) == "status-warning"

    def test_latency_class_bad(self) -> None:
        """Test latency >= 150ms returns 'status-critical' class."""
        assert PerformanceDashboardWidget._get_latency_class(150) == "status-critical"
        assert PerformanceDashboardWidget._get_latency_class(200) == "status-critical"
        assert PerformanceDashboardWidget._get_latency_class(1000) == "status-critical"

    def test_memory_class_good(self) -> None:
        """Test memory < 60% returns 'status-ok' class."""
        assert PerformanceDashboardWidget._get_memory_class(0) == "status-ok"
        assert PerformanceDashboardWidget._get_memory_class(30) == "status-ok"
        assert PerformanceDashboardWidget._get_memory_class(59) == "status-ok"

    def test_memory_class_warning(self) -> None:
        """Test memory between 60-80% returns 'status-warning' class."""
        assert PerformanceDashboardWidget._get_memory_class(60) == "status-warning"
        assert PerformanceDashboardWidget._get_memory_class(70) == "status-warning"
        assert PerformanceDashboardWidget._get_memory_class(79) == "status-warning"

    def test_memory_class_bad(self) -> None:
        """Test memory >= 80% returns 'status-critical' class."""
        assert PerformanceDashboardWidget._get_memory_class(80) == "status-critical"
        assert PerformanceDashboardWidget._get_memory_class(90) == "status-critical"
        assert PerformanceDashboardWidget._get_memory_class(100) == "status-critical"

    def test_cpu_class_good(self) -> None:
        """Test CPU < 50% returns 'status-ok' class."""
        assert PerformanceDashboardWidget._get_cpu_class(0) == "status-ok"
        assert PerformanceDashboardWidget._get_cpu_class(25) == "status-ok"
        assert PerformanceDashboardWidget._get_cpu_class(49) == "status-ok"

    def test_cpu_class_warning(self) -> None:
        """Test CPU between 50-80% returns 'status-warning' class."""
        assert PerformanceDashboardWidget._get_cpu_class(50) == "status-warning"
        assert PerformanceDashboardWidget._get_cpu_class(65) == "status-warning"
        assert PerformanceDashboardWidget._get_cpu_class(79) == "status-warning"

    def test_cpu_class_bad(self) -> None:
        """Test CPU >= 80% returns 'status-critical' class."""
        assert PerformanceDashboardWidget._get_cpu_class(80) == "status-critical"
        assert PerformanceDashboardWidget._get_cpu_class(90) == "status-critical"
        assert PerformanceDashboardWidget._get_cpu_class(100) == "status-critical"

    def test_watch_snapshot_none(self) -> None:
        """Test that watch_snapshot does nothing with None."""
        widget = PerformanceDashboardWidget()
        widget._fps_label = MagicMock()
        widget._latency_label = MagicMock()
        widget.watch_snapshot(None)
        widget._fps_label.update.assert_not_called()
        widget._latency_label.update.assert_not_called()

    def test_watch_snapshot_updates_cached_labels(self) -> None:
        """Test that watch_snapshot updates labels when cached references exist."""
        widget = PerformanceDashboardWidget()

        # Mock cached label references
        mock_fps_label = MagicMock()
        mock_latency_label = MagicMock()
        mock_memory_label = MagicMock()
        mock_budget_label = MagicMock()

        widget._fps_label = mock_fps_label
        widget._latency_label = mock_latency_label
        widget._memory_label = mock_memory_label
        widget._budget_label = mock_budget_label

        snapshot = PerformanceSnapshot(
            fps=1.0,
            avg_frame_time=0.030,
            p95_frame_time=0.050,
            memory_mb=256.0,
            memory_percent=25.0,
        )

        widget.watch_snapshot(snapshot)

        # Verify FPS label was updated (with icon prefix)
        mock_fps_label.update.assert_called_once_with("✓ 1.0")
        mock_fps_label.set_classes.assert_called_once_with("perf-value status-ok")

        # Verify latency label was updated (30ms avg, 50ms p95, with icon prefix)
        mock_latency_label.update.assert_called_once_with("✓ 30ms avg, 50ms p95")
        mock_latency_label.set_classes.assert_called_once_with("perf-value status-ok")

        # Verify memory label was updated (with icon prefix)
        mock_memory_label.update.assert_called_once_with("✓ 256MB (25%)")
        mock_memory_label.set_classes.assert_called_once_with("perf-value status-ok")

    def test_watch_snapshot_applies_warning_classes(self) -> None:
        """Test that watch_snapshot applies warning classes for degraded metrics."""
        widget = PerformanceDashboardWidget()

        mock_fps_label = MagicMock()
        mock_latency_label = MagicMock()
        mock_memory_label = MagicMock()
        mock_budget_label = MagicMock()

        widget._fps_label = mock_fps_label
        widget._latency_label = mock_latency_label
        widget._memory_label = mock_memory_label
        widget._budget_label = mock_budget_label

        snapshot = PerformanceSnapshot(
            fps=0.3,  # warning: 0.2-0.5
            avg_frame_time=0.075,  # warning: 50-150ms
            p95_frame_time=0.090,
            memory_mb=512.0,
            memory_percent=70.0,  # warning: 60-80%
        )

        widget.watch_snapshot(snapshot)

        mock_fps_label.set_classes.assert_called_once_with("perf-value status-warning")
        mock_latency_label.set_classes.assert_called_once_with("perf-value status-warning")
        mock_memory_label.set_classes.assert_called_once_with("perf-value status-warning")

    def test_watch_snapshot_applies_bad_classes(self) -> None:
        """Test that watch_snapshot applies critical classes for poor metrics."""
        widget = PerformanceDashboardWidget()

        mock_fps_label = MagicMock()
        mock_latency_label = MagicMock()
        mock_memory_label = MagicMock()
        mock_budget_label = MagicMock()

        widget._fps_label = mock_fps_label
        widget._latency_label = mock_latency_label
        widget._memory_label = mock_memory_label
        widget._budget_label = mock_budget_label

        snapshot = PerformanceSnapshot(
            fps=0.1,  # critical: <0.2
            avg_frame_time=0.200,  # critical: >=150ms
            p95_frame_time=0.250,
            memory_mb=800.0,
            memory_percent=85.0,  # critical: >=80%
        )

        widget.watch_snapshot(snapshot)

        mock_fps_label.set_classes.assert_called_once_with("perf-value status-critical")
        mock_latency_label.set_classes.assert_called_once_with("perf-value status-critical")
        mock_memory_label.set_classes.assert_called_once_with("perf-value status-critical")

    def test_refresh_metrics_gets_snapshot(self) -> None:
        """Test that _refresh_metrics fetches snapshot from service."""
        widget = PerformanceDashboardWidget()

        # Mock cached label references needed by watch_snapshot
        widget._fps_label = MagicMock()
        widget._latency_label = MagicMock()
        widget._memory_label = MagicMock()
        widget._budget_label = MagicMock()

        mock_snapshot = PerformanceSnapshot(fps=1.0)

        with patch(
            "gpt_trader.tui.widgets.performance_dashboard.get_tui_performance_service"
        ) as mock_get_service:
            mock_service = MagicMock()
            mock_service.get_snapshot.return_value = mock_snapshot
            mock_get_service.return_value = mock_service

            widget._refresh_metrics()

            mock_service.get_snapshot.assert_called_once()
            assert widget.snapshot == mock_snapshot

    def test_compact_mode_skips_expanded_metrics(self) -> None:
        """Test that compact mode doesn't update expanded-only metrics."""
        widget = PerformanceDashboardWidget(compact=True)

        mock_fps_label = MagicMock()
        mock_latency_label = MagicMock()
        mock_memory_label = MagicMock()
        mock_budget_label = MagicMock()

        widget._fps_label = mock_fps_label
        widget._latency_label = mock_latency_label
        widget._memory_label = mock_memory_label
        widget._budget_label = mock_budget_label
        widget._throttler_label = None  # Not set in compact mode

        snapshot = PerformanceSnapshot(
            fps=1.0,
            avg_frame_time=0.030,
            p95_frame_time=0.050,
            memory_mb=256.0,
            memory_percent=25.0,
            throttler_batch_size=2.5,
            throttler_queue_depth=5,
        )

        # Should not raise even with throttler stats present
        widget.watch_snapshot(snapshot)

        # Basic metrics should still update
        mock_fps_label.update.assert_called_once()
        mock_latency_label.update.assert_called_once()
        mock_memory_label.update.assert_called_once()


class TestPerformanceDashboardThresholds:
    """Tests verifying threshold boundary conditions."""

    @pytest.mark.parametrize(
        "fps,expected_class",
        [
            (0.5, "status-ok"),  # exactly at ok threshold
            (0.4999, "status-warning"),  # just below ok
            (0.2, "status-warning"),  # exactly at warning threshold
            (0.1999, "status-critical"),  # just below warning
        ],
    )
    def test_fps_threshold_boundaries(self, fps: float, expected_class: str) -> None:
        """Test FPS threshold boundaries."""
        assert PerformanceDashboardWidget._get_fps_class(fps) == expected_class

    @pytest.mark.parametrize(
        "latency_ms,expected_class",
        [
            (49.9, "status-ok"),  # just below warning
            (50, "status-warning"),  # exactly at warning threshold
            (149.9, "status-warning"),  # just below critical
            (150, "status-critical"),  # exactly at critical threshold
        ],
    )
    def test_latency_threshold_boundaries(self, latency_ms: float, expected_class: str) -> None:
        """Test latency threshold boundaries."""
        assert PerformanceDashboardWidget._get_latency_class(latency_ms) == expected_class

    @pytest.mark.parametrize(
        "memory_percent,expected_class",
        [
            (59.9, "status-ok"),  # just below warning
            (60, "status-warning"),  # exactly at warning threshold
            (79.9, "status-warning"),  # just below critical
            (80, "status-critical"),  # exactly at critical threshold
        ],
    )
    def test_memory_threshold_boundaries(self, memory_percent: float, expected_class: str) -> None:
        """Test memory threshold boundaries."""
        assert PerformanceDashboardWidget._get_memory_class(memory_percent) == expected_class

    @pytest.mark.parametrize(
        "cpu_percent,expected_class",
        [
            (49.9, "status-ok"),  # just below warning
            (50, "status-warning"),  # exactly at warning threshold
            (79.9, "status-warning"),  # just below critical
            (80, "status-critical"),  # exactly at critical threshold
        ],
    )
    def test_cpu_threshold_boundaries(self, cpu_percent: float, expected_class: str) -> None:
        """Test CPU threshold boundaries."""
        assert PerformanceDashboardWidget._get_cpu_class(cpu_percent) == expected_class
