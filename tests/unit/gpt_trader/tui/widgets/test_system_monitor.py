"""Tests for SystemMonitorWidget initial state, updates, and resilience with partial data."""

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


class TestSystemMonitorPartialState:
    """Tests for widget resilience with partial/missing state data."""

    @pytest.mark.asyncio
    async def test_handles_partial_state_system_only(self) -> None:
        """Test that widget handles state with only system_data set."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # State with only system_data, no resilience or execution data
            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = SystemStatus(
                cpu_usage="50%",
                api_latency=100.0,
            )
            # Explicitly set others to None/defaults
            state.resilience_data = None
            state.execution_data = None

            # Should not raise
            widget.on_state_updated(state)

            # System values should be extracted
            assert widget.cpu_usage == 50.0
            assert widget.latency == 100.0

    @pytest.mark.asyncio
    async def test_handles_degraded_mode_with_no_data(self) -> None:
        """Test that widget handles degraded mode with missing data sections."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.degraded_mode = True
            state.degraded_reason = "StatusReporter unavailable"
            state.system_data = SystemStatus()  # Empty but not None

            # Should not raise
            widget.on_state_updated(state)

            # Widget should still be functional with defaults
            assert widget.cpu_usage == 0.0

    @pytest.mark.asyncio
    async def test_handles_connection_unhealthy_state(self) -> None:
        """Test that widget handles unhealthy connection states."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.connection_healthy = False
            state.running = False
            state.data_source_mode = "live"  # Non-demo mode triggers STOPPED
            state.system_data = SystemStatus(
                connection_status="DISCONNECTED",
            )

            # Should not raise
            widget.on_state_updated(state)

            # Connection status should reflect stopped when not running in non-demo mode
            assert widget.connection_status == "STOPPED"

    @pytest.mark.asyncio
    async def test_handles_resilience_data_with_no_last_update(self) -> None:
        """Test that widget handles resilience data with last_update=0."""
        from gpt_trader.tui.types import ResilienceState

        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = SystemStatus(cpu_usage="25%")
            state.resilience_data = ResilienceState(
                latency_p50_ms=50.0,
                latency_p95_ms=100.0,
                error_rate=0.01,
                cache_hit_rate=0.8,
                any_circuit_open=False,
                last_update=0,  # Not yet initialized
            )

            # Should not raise and should NOT update resilience metrics
            # (last_update=0 means no valid data yet)
            widget.on_state_updated(state)

            # Resilience values should remain at defaults
            assert widget.latency_p50 == 0.0
            assert widget.latency_p95 == 0.0
