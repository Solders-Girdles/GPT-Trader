"""Tests for PerformanceDashboardWidget helper classification logic."""

from __future__ import annotations

import pytest

from gpt_trader.tui.widgets.performance_dashboard import PerformanceDashboardWidget


class TestPerformanceDashboardWidgetInit:
    def test_init_default_mode(self) -> None:
        """Test widget initializes in default (expanded) mode."""
        widget = PerformanceDashboardWidget()

        assert widget._compact is False
        assert widget._refresh_timer is None

    def test_init_compact_mode(self) -> None:
        """Test widget initializes in compact mode."""
        widget = PerformanceDashboardWidget(compact=True)

        assert widget._compact is True


class TestPerformanceDashboardMetricClasses:
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


class TestPerformanceDashboardThresholds:
    """Tests verifying threshold boundary conditions."""

    @pytest.mark.parametrize(
        "fps,expected_class",
        [
            (0.5, "status-ok"),
            (0.4999, "status-warning"),
            (0.2, "status-warning"),
            (0.1999, "status-critical"),
        ],
    )
    def test_fps_threshold_boundaries(self, fps: float, expected_class: str) -> None:
        assert PerformanceDashboardWidget._get_fps_class(fps) == expected_class

    @pytest.mark.parametrize(
        "latency_ms,expected_class",
        [
            (49.9, "status-ok"),
            (50, "status-warning"),
            (149.9, "status-warning"),
            (150, "status-critical"),
        ],
    )
    def test_latency_threshold_boundaries(self, latency_ms: float, expected_class: str) -> None:
        assert PerformanceDashboardWidget._get_latency_class(latency_ms) == expected_class

    @pytest.mark.parametrize(
        "memory_percent,expected_class",
        [
            (59.9, "status-ok"),
            (60, "status-warning"),
            (79.9, "status-warning"),
            (80, "status-critical"),
        ],
    )
    def test_memory_threshold_boundaries(self, memory_percent: float, expected_class: str) -> None:
        assert PerformanceDashboardWidget._get_memory_class(memory_percent) == expected_class

    @pytest.mark.parametrize(
        "cpu_percent,expected_class",
        [
            (49.9, "status-ok"),
            (50, "status-warning"),
            (79.9, "status-warning"),
            (80, "status-critical"),
        ],
    )
    def test_cpu_threshold_boundaries(self, cpu_percent: float, expected_class: str) -> None:
        assert PerformanceDashboardWidget._get_cpu_class(cpu_percent) == expected_class
