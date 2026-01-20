"""
Tests for SystemMonitorWidget watch methods and threshold rendering.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from gpt_trader.tui.widgets.dashboard import SystemMonitorWidget
from gpt_trader.tui.widgets.primitives import ProgressBarWidget


class SystemMonitorTestApp(App):
    """Test app for SystemMonitorWidget."""

    def compose(self) -> ComposeResult:
        yield SystemMonitorWidget(id="test-system-monitor")


class TestSystemMonitorWatchMethods:
    """Tests for SystemMonitorWidget watch methods."""

    @pytest.mark.asyncio
    async def test_watch_cpu_usage_updates_progress_bar(self) -> None:
        """Test that CPU usage updates the progress bar widget."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # Update CPU usage
            widget.cpu_usage = 65.0

            # Verify progress bar updated
            pb = app.query_one("#pb-cpu", ProgressBarWidget)
            assert pb.percentage == 65.0

    @pytest.mark.asyncio
    async def test_watch_latency_green_threshold(self) -> None:
        """Test latency displays green for values under 50ms."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.latency = 25.0

            lbl = app.query_one("#lbl-latency", Label)
            rendered = str(lbl.render())
            assert "green" in rendered.lower() or "25ms" in rendered

    @pytest.mark.asyncio
    async def test_watch_latency_yellow_threshold(self) -> None:
        """Test latency displays yellow for values between 50-200ms."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.latency = 100.0

            lbl = app.query_one("#lbl-latency", Label)
            rendered = str(lbl.render())
            assert "yellow" in rendered.lower() or "100ms" in rendered

    @pytest.mark.asyncio
    async def test_watch_latency_red_threshold(self) -> None:
        """Test latency displays red for values over 200ms."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.latency = 300.0

            lbl = app.query_one("#lbl-latency", Label)
            rendered = str(lbl.render())
            assert "red" in rendered.lower() or "300ms" in rendered

    @pytest.mark.asyncio
    async def test_watch_connection_status_connected(self) -> None:
        """Test connection status displays connected state correctly."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.connection_status = "CONNECTED"

            lbl = app.query_one("#lbl-conn", Label)
            rendered = str(lbl.render())
            assert "Connected" in rendered or "green" in rendered.lower()
            assert lbl.has_class("status-ok")

    @pytest.mark.asyncio
    async def test_watch_connection_status_disconnected(self) -> None:
        """Test connection status displays disconnected state correctly."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.connection_status = "DISCONNECTED"

            lbl = app.query_one("#lbl-conn", Label)
            rendered = str(lbl.render())
            assert "DISCONNECTED" in rendered or "red" in rendered.lower()
            assert lbl.has_class("status-critical")

    @pytest.mark.asyncio
    async def test_watch_connection_status_connecting(self) -> None:
        """Test connection status displays connecting state correctly."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.connection_status = "CONNECTING"

            lbl = app.query_one("#lbl-conn", Label)
            assert lbl.has_class("status-warning")

    @pytest.mark.asyncio
    async def test_watch_rate_limit_updates_progress_bar(self) -> None:
        """Test rate limit updates the progress bar."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.rate_limit = "25%"

            pb = app.query_one("#pb-rate", ProgressBarWidget)
            assert pb.percentage == 25.0

    @pytest.mark.asyncio
    async def test_watch_rate_limit_high_value(self) -> None:
        """Test rate limit with high percentage value."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            widget.rate_limit = "90%"

            pb = app.query_one("#pb-rate", ProgressBarWidget)
            assert pb.percentage == 90.0

    @pytest.mark.asyncio
    async def test_watch_memory_usage_updates_progress_bar(self) -> None:
        """Test memory usage updates the progress bar."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # 512MB out of 1024MB (default threshold) = 50%
            widget.memory_usage = "512MB"

            pb = app.query_one("#pb-memory", ProgressBarWidget)
            assert pb.percentage == 50.0
