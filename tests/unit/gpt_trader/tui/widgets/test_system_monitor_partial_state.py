"""Tests for SystemMonitorWidget resilience with partial/missing state data."""

import pytest
from textual.app import App, ComposeResult

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import SystemStatus
from gpt_trader.tui.widgets.dashboard import SystemMonitorWidget


class SystemMonitorTestApp(App):
    """Test app for SystemMonitorWidget."""

    def compose(self) -> ComposeResult:
        yield SystemMonitorWidget(id="test-system-monitor")


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
