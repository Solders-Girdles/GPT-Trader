"""
Tests for SystemMonitorWidget initial state and state updates.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Label

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import SystemStatus
from gpt_trader.tui.widgets.dashboard import SystemMonitorWidget


class SystemMonitorTestApp(App):
    """Test app for SystemMonitorWidget."""

    def compose(self) -> ComposeResult:
        yield SystemMonitorWidget(id="test-system-monitor")


class TestSystemMonitorWidgetInitialStateAndUpdates:
    """Tests for SystemMonitorWidget defaults and state extraction."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """Test that widget renders with default values."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # Check initial reactive property values
            assert widget.cpu_usage == 0.0
            assert widget.memory_usage == "0MB"
            assert widget.latency == 0.0
            assert widget.connection_status == "CONNECTING"
            assert widget.rate_limit == "0%"

            # Check that header is rendered
            header = app.query_one(".sys-header", Label)
            assert "SYSTEM" in str(header.render())

    @pytest.mark.asyncio
    async def test_on_state_updated_extracts_system_data(self) -> None:
        """Test that on_state_updated correctly extracts system data."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # Create state with system data
            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = SystemStatus(
                api_latency=123.45,
                connection_status="CONNECTED",
                rate_limit_usage="15%",
                memory_usage="256MB",
                cpu_usage="45%",
            )

            # Trigger state update
            widget.on_state_updated(state)

            # Verify extracted values
            assert widget.cpu_usage == 45.0
            assert widget.latency == 123.45
            assert widget.memory_usage == "256MB"
            assert widget.connection_status == "CONNECTED"
            assert widget.rate_limit == "15%"

    @pytest.mark.asyncio
    async def test_on_state_updated_handles_numeric_cpu(self) -> None:
        """Test that on_state_updated handles numeric CPU values."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = SystemStatus(
                cpu_usage=75.5,  # Numeric value, not string
            )

            widget.on_state_updated(state)
            assert widget.cpu_usage == 75.5

    @pytest.mark.asyncio
    async def test_on_state_updated_handles_missing_system_data(self) -> None:
        """Test that on_state_updated gracefully handles missing system data."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # State with no system_data
            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = None

            # Should not raise
            widget.on_state_updated(state)

            # Values should remain at defaults
            assert widget.cpu_usage == 0.0
