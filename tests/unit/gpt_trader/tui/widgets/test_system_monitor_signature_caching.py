"""Tests for SystemMonitorWidget display signature caching optimization."""

import pytest
from textual.app import App, ComposeResult

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import SystemStatus
from gpt_trader.tui.widgets.dashboard import SystemMonitorWidget


class SystemMonitorTestApp(App):
    """Test app for SystemMonitorWidget."""

    def compose(self) -> ComposeResult:
        yield SystemMonitorWidget(id="test-system-monitor")


class TestSystemMonitorSignatureCaching:
    """Tests for display signature caching optimization."""

    @pytest.mark.asyncio
    async def test_skips_update_when_signature_unchanged(self) -> None:
        """Test that on_state_updated skips work when state unchanged."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # Create identical states
            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = SystemStatus(
                cpu_usage="50%",
                api_latency=100.0,
            )

            # First call should update
            widget.on_state_updated(state)
            assert widget.cpu_usage == 50.0

            # Track that signature is cached
            cached_sig = widget._last_display_signature
            assert cached_sig is not None

            # Second call with same state should hit cache
            widget.on_state_updated(state)

            # Signature should be the same (early exit)
            assert widget._last_display_signature == cached_sig

    @pytest.mark.asyncio
    async def test_updates_when_single_field_changes(self) -> None:
        """Test that on_state_updated runs when any field changes."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            # Initial state
            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = SystemStatus(
                cpu_usage="50%",
                api_latency=100.0,
            )
            widget.on_state_updated(state)
            original_sig = widget._last_display_signature

            # Change only one field
            state.system_data = SystemStatus(
                cpu_usage="75%",  # Changed
                api_latency=100.0,  # Same
            )
            widget.on_state_updated(state)

            # Signature should be different
            assert widget._last_display_signature != original_sig
            assert widget.cpu_usage == 75.0

    @pytest.mark.asyncio
    async def test_signature_includes_running_state(self) -> None:
        """Test that signature changes when running state changes."""
        app = SystemMonitorTestApp()

        async with app.run_test():
            widget = app.query_one(SystemMonitorWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.system_data = SystemStatus(cpu_usage="50%")
            state.running = True

            widget.on_state_updated(state)
            sig_running = widget._last_display_signature

            # Change only running state
            state.running = False
            widget.on_state_updated(state)

            assert widget._last_display_signature != sig_running
