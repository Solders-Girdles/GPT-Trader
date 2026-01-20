"""Tests for PerformanceDashboardWidget snapshot rendering/refresh."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.tui.widgets.performance_dashboard as perf_dashboard_module
from gpt_trader.tui.services.performance_service import PerformanceSnapshot
from gpt_trader.tui.widgets.performance_dashboard import PerformanceDashboardWidget


class TestPerformanceDashboardWidgetWatchSnapshot:
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

        mock_fps_label.update.assert_called_once_with("✓ 1.0")
        mock_fps_label.set_classes.assert_called_once_with("perf-value status-ok")

        mock_latency_label.update.assert_called_once_with("✓ 30ms avg, 50ms p95")
        mock_latency_label.set_classes.assert_called_once_with("perf-value status-ok")

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
            fps=0.3,
            avg_frame_time=0.075,
            p95_frame_time=0.090,
            memory_mb=512.0,
            memory_percent=70.0,
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
            fps=0.1,
            avg_frame_time=0.200,
            p95_frame_time=0.250,
            memory_mb=800.0,
            memory_percent=85.0,
        )

        widget.watch_snapshot(snapshot)

        mock_fps_label.set_classes.assert_called_once_with("perf-value status-critical")
        mock_latency_label.set_classes.assert_called_once_with("perf-value status-critical")
        mock_memory_label.set_classes.assert_called_once_with("perf-value status-critical")

    def test_refresh_metrics_gets_snapshot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that _refresh_metrics fetches snapshot from service."""
        widget = PerformanceDashboardWidget()

        widget._fps_label = MagicMock()
        widget._latency_label = MagicMock()
        widget._memory_label = MagicMock()
        widget._budget_label = MagicMock()

        mock_snapshot = PerformanceSnapshot(fps=1.0)
        mock_service = MagicMock()
        mock_service.get_snapshot.return_value = mock_snapshot
        monkeypatch.setattr(
            perf_dashboard_module, "get_tui_performance_service", lambda: mock_service
        )

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
        widget._throttler_label = None

        snapshot = PerformanceSnapshot(
            fps=1.0,
            avg_frame_time=0.030,
            p95_frame_time=0.050,
            memory_mb=256.0,
            memory_percent=25.0,
            throttler_batch_size=2.5,
            throttler_queue_depth=5,
        )

        widget.watch_snapshot(snapshot)

        mock_fps_label.update.assert_called_once()
        mock_latency_label.update.assert_called_once()
        mock_memory_label.update.assert_called_once()
