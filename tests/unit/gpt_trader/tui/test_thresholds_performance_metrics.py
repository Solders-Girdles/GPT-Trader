"""Tests for TUI performance threshold functions."""

from gpt_trader.tui.thresholds import (
    StatusLevel,
    get_cpu_status,
    get_latency_status,
    get_memory_status,
)


class TestPerformanceThresholds:
    """Tests for performance metric thresholds."""

    def test_latency_ok(self):
        """Low latency returns OK status."""
        assert get_latency_status(30.0) == StatusLevel.OK

    def test_latency_warning(self):
        """Moderate latency returns WARNING status."""
        assert get_latency_status(100.0) == StatusLevel.WARNING

    def test_latency_critical(self):
        """High latency returns CRITICAL status."""
        assert get_latency_status(200.0) == StatusLevel.CRITICAL

    def test_cpu_ok(self):
        """Low CPU returns OK status."""
        assert get_cpu_status(30.0) == StatusLevel.OK

    def test_cpu_warning(self):
        """Moderate CPU returns WARNING status."""
        assert get_cpu_status(65.0) == StatusLevel.WARNING

    def test_cpu_critical(self):
        """High CPU returns CRITICAL status."""
        assert get_cpu_status(90.0) == StatusLevel.CRITICAL

    def test_memory_ok(self):
        """Low memory returns OK status."""
        assert get_memory_status(40.0) == StatusLevel.OK

    def test_memory_warning(self):
        """Moderate memory returns WARNING status."""
        assert get_memory_status(70.0) == StatusLevel.WARNING

    def test_memory_critical(self):
        """High memory returns CRITICAL status."""
        assert get_memory_status(90.0) == StatusLevel.CRITICAL
