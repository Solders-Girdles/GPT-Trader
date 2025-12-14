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
        """Test FPS >= 0.5 returns 'good' class."""
        assert PerformanceDashboardWidget._get_fps_class(0.5) == "good"
        assert PerformanceDashboardWidget._get_fps_class(1.0) == "good"
        assert PerformanceDashboardWidget._get_fps_class(10.0) == "good"

    def test_fps_class_warning(self) -> None:
        """Test FPS between 0.2 and 0.5 returns 'warning' class."""
        assert PerformanceDashboardWidget._get_fps_class(0.2) == "warning"
        assert PerformanceDashboardWidget._get_fps_class(0.3) == "warning"
        assert PerformanceDashboardWidget._get_fps_class(0.49) == "warning"

    def test_fps_class_bad(self) -> None:
        """Test FPS < 0.2 returns 'bad' class."""
        assert PerformanceDashboardWidget._get_fps_class(0.0) == "bad"
        assert PerformanceDashboardWidget._get_fps_class(0.1) == "bad"
        assert PerformanceDashboardWidget._get_fps_class(0.19) == "bad"

    def test_latency_class_good(self) -> None:
        """Test latency < 50ms returns 'good' class."""
        assert PerformanceDashboardWidget._get_latency_class(0) == "good"
        assert PerformanceDashboardWidget._get_latency_class(25) == "good"
        assert PerformanceDashboardWidget._get_latency_class(49) == "good"

    def test_latency_class_warning(self) -> None:
        """Test latency between 50-100ms returns 'warning' class."""
        assert PerformanceDashboardWidget._get_latency_class(50) == "warning"
        assert PerformanceDashboardWidget._get_latency_class(75) == "warning"
        assert PerformanceDashboardWidget._get_latency_class(99) == "warning"

    def test_latency_class_bad(self) -> None:
        """Test latency >= 100ms returns 'bad' class."""
        assert PerformanceDashboardWidget._get_latency_class(100) == "bad"
        assert PerformanceDashboardWidget._get_latency_class(200) == "bad"
        assert PerformanceDashboardWidget._get_latency_class(1000) == "bad"

    def test_memory_class_good(self) -> None:
        """Test memory < 50% returns 'good' class."""
        assert PerformanceDashboardWidget._get_memory_class(0) == "good"
        assert PerformanceDashboardWidget._get_memory_class(25) == "good"
        assert PerformanceDashboardWidget._get_memory_class(49) == "good"

    def test_memory_class_warning(self) -> None:
        """Test memory between 50-80% returns 'warning' class."""
        assert PerformanceDashboardWidget._get_memory_class(50) == "warning"
        assert PerformanceDashboardWidget._get_memory_class(65) == "warning"
        assert PerformanceDashboardWidget._get_memory_class(79) == "warning"

    def test_memory_class_bad(self) -> None:
        """Test memory >= 80% returns 'bad' class."""
        assert PerformanceDashboardWidget._get_memory_class(80) == "bad"
        assert PerformanceDashboardWidget._get_memory_class(90) == "bad"
        assert PerformanceDashboardWidget._get_memory_class(100) == "bad"

    def test_cpu_class_good(self) -> None:
        """Test CPU < 50% returns 'good' class."""
        assert PerformanceDashboardWidget._get_cpu_class(0) == "good"
        assert PerformanceDashboardWidget._get_cpu_class(25) == "good"
        assert PerformanceDashboardWidget._get_cpu_class(49) == "good"

    def test_cpu_class_warning(self) -> None:
        """Test CPU between 50-80% returns 'warning' class."""
        assert PerformanceDashboardWidget._get_cpu_class(50) == "warning"
        assert PerformanceDashboardWidget._get_cpu_class(65) == "warning"
        assert PerformanceDashboardWidget._get_cpu_class(79) == "warning"

    def test_cpu_class_bad(self) -> None:
        """Test CPU >= 80% returns 'bad' class."""
        assert PerformanceDashboardWidget._get_cpu_class(80) == "bad"
        assert PerformanceDashboardWidget._get_cpu_class(90) == "bad"
        assert PerformanceDashboardWidget._get_cpu_class(100) == "bad"

    def test_watch_snapshot_none(self) -> None:
        """Test that watch_snapshot does nothing with None."""
        widget = PerformanceDashboardWidget()
        # Should not raise
        widget.watch_snapshot(None)

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

        # Verify FPS label was updated
        mock_fps_label.update.assert_called_once_with("1.0")
        mock_fps_label.set_classes.assert_called_once_with("perf-value good")

        # Verify latency label was updated (30ms avg, 50ms p95)
        mock_latency_label.update.assert_called_once_with("30ms avg, 50ms p95")
        mock_latency_label.set_classes.assert_called_once_with("perf-value good")

        # Verify memory label was updated
        mock_memory_label.update.assert_called_once_with("256MB (25%)")
        mock_memory_label.set_classes.assert_called_once_with("perf-value good")

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
            avg_frame_time=0.075,  # warning: 50-100ms
            p95_frame_time=0.090,
            memory_mb=512.0,
            memory_percent=65.0,  # warning: 50-80%
        )

        widget.watch_snapshot(snapshot)

        mock_fps_label.set_classes.assert_called_once_with("perf-value warning")
        mock_latency_label.set_classes.assert_called_once_with("perf-value warning")
        mock_memory_label.set_classes.assert_called_once_with("perf-value warning")

    def test_watch_snapshot_applies_bad_classes(self) -> None:
        """Test that watch_snapshot applies bad classes for poor metrics."""
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
            fps=0.1,  # bad: <0.2
            avg_frame_time=0.150,  # bad: >=100ms
            p95_frame_time=0.200,
            memory_mb=800.0,
            memory_percent=85.0,  # bad: >=80%
        )

        widget.watch_snapshot(snapshot)

        mock_fps_label.set_classes.assert_called_once_with("perf-value bad")
        mock_latency_label.set_classes.assert_called_once_with("perf-value bad")
        mock_memory_label.set_classes.assert_called_once_with("perf-value bad")

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
            (0.5, "good"),  # exactly at good threshold
            (0.4999, "warning"),  # just below good
            (0.2, "warning"),  # exactly at warning threshold
            (0.1999, "bad"),  # just below warning
        ],
    )
    def test_fps_threshold_boundaries(self, fps: float, expected_class: str) -> None:
        """Test FPS threshold boundaries."""
        assert PerformanceDashboardWidget._get_fps_class(fps) == expected_class

    @pytest.mark.parametrize(
        "latency_ms,expected_class",
        [
            (49.9, "good"),  # just below warning
            (50, "warning"),  # exactly at warning threshold
            (99.9, "warning"),  # just below bad
            (100, "bad"),  # exactly at bad threshold
        ],
    )
    def test_latency_threshold_boundaries(
        self, latency_ms: float, expected_class: str
    ) -> None:
        """Test latency threshold boundaries."""
        assert PerformanceDashboardWidget._get_latency_class(latency_ms) == expected_class

    @pytest.mark.parametrize(
        "memory_percent,expected_class",
        [
            (49.9, "good"),  # just below warning
            (50, "warning"),  # exactly at warning threshold
            (79.9, "warning"),  # just below bad
            (80, "bad"),  # exactly at bad threshold
        ],
    )
    def test_memory_threshold_boundaries(
        self, memory_percent: float, expected_class: str
    ) -> None:
        """Test memory threshold boundaries."""
        assert (
            PerformanceDashboardWidget._get_memory_class(memory_percent) == expected_class
        )

    @pytest.mark.parametrize(
        "cpu_percent,expected_class",
        [
            (49.9, "good"),  # just below warning
            (50, "warning"),  # exactly at warning threshold
            (79.9, "warning"),  # just below bad
            (80, "bad"),  # exactly at bad threshold
        ],
    )
    def test_cpu_threshold_boundaries(
        self, cpu_percent: float, expected_class: str
    ) -> None:
        """Test CPU threshold boundaries."""
        assert PerformanceDashboardWidget._get_cpu_class(cpu_percent) == expected_class
