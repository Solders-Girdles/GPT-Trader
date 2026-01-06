"""
Tests for PositionCardWidget in dashboard_position.py.

Tests cover:
- Partial state handling (missing data sections)
- Degraded mode and connection unhealthy states
- Display signature caching optimization
- Banner logic for various conditions
"""

from decimal import Decimal

import pytest
from textual.app import App, ComposeResult

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import PortfolioSummary, Position, SystemStatus
from gpt_trader.tui.widgets.dashboard_position import PositionCardWidget


class PositionCardTestApp(App):
    """Test app for PositionCardWidget."""

    def compose(self) -> ComposeResult:
        yield PositionCardWidget(id="test-position-card")


class TestPositionCardWidget:
    """Tests for PositionCardWidget basic functionality."""

    @pytest.mark.asyncio
    async def test_initial_state(self) -> None:
        """Test that widget renders in loading state initially."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            # Initial state should not have received update
            assert not widget._has_received_update

    @pytest.mark.asyncio
    async def test_on_state_updated_with_position(self) -> None:
        """Test that widget correctly displays position data."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.position_data = PortfolioSummary(
                positions={
                    "BTC-USD": Position(
                        symbol="BTC-USD",
                        quantity=Decimal("0.1"),
                        side="LONG",
                        leverage=2,
                        unrealized_pnl=Decimal("500.00"),
                        entry_price=Decimal("95000.00"),
                        mark_price=Decimal("96000.00"),
                        liquidation_price=Decimal("80000.00"),
                    )
                }
            )
            state.system_data = SystemStatus(connection_status="CONNECTED")

            widget.on_state_updated(state)

            assert widget._has_received_update
            assert widget.position_data is not None
            assert widget.position_data["symbol"] == "BTC-USD"


class TestPositionCardPartialState:
    """Tests for widget resilience with partial/missing state data."""

    @pytest.mark.asyncio
    async def test_handles_no_position_data(self) -> None:
        """Test that widget handles state with no positions."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.position_data = PortfolioSummary(positions={})
            state.system_data = SystemStatus(connection_status="CONNECTED")

            # Should not raise
            widget.on_state_updated(state)

            assert widget._has_received_update
            assert widget.position_data is None

    @pytest.mark.asyncio
    async def test_handles_missing_system_data(self) -> None:
        """Test that widget handles state with no system_data."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.system_data = None
            state.position_data = PortfolioSummary(
                positions={
                    "ETH-USD": Position(
                        symbol="ETH-USD",
                        quantity=Decimal("1.0"),
                        side="SHORT",
                        leverage=1,
                        unrealized_pnl=Decimal("-100.00"),
                        entry_price=Decimal("3500.00"),
                        mark_price=Decimal("3550.00"),
                    )
                }
            )

            # Should not raise
            widget.on_state_updated(state)

            assert widget._connection_status == ""

    @pytest.mark.asyncio
    async def test_handles_degraded_mode(self) -> None:
        """Test that widget handles degraded mode with staleness banner."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.degraded_mode = True
            state.degraded_reason = "StatusReporter unavailable"
            state.running = True
            state.position_data = PortfolioSummary(positions={})
            state.system_data = SystemStatus(connection_status="CONNECTED")

            # Should not raise
            widget.on_state_updated(state)

            assert widget._has_received_update

    @pytest.mark.asyncio
    async def test_handles_connection_unhealthy(self) -> None:
        """Test that widget handles unhealthy connection states."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.connection_healthy = False
            state.running = True
            state.system_data = SystemStatus(connection_status="DISCONNECTED")
            state.position_data = PortfolioSummary(positions={})

            # Should not raise
            widget.on_state_updated(state)

            assert widget._connection_status == "DISCONNECTED"

    @pytest.mark.asyncio
    async def test_handles_bot_not_running(self) -> None:
        """Test that widget handles stopped bot state."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = False
            state.system_data = SystemStatus(connection_status="IDLE")
            state.position_data = PortfolioSummary(positions={})

            # Should not raise
            widget.on_state_updated(state)

            assert not widget._bot_running


class TestPositionCardSignatureCaching:
    """Tests for display signature caching optimization."""

    @pytest.mark.asyncio
    async def test_skips_update_when_signature_unchanged(self) -> None:
        """Test that on_state_updated skips work when state unchanged."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.system_data = SystemStatus(connection_status="CONNECTED")
            state.position_data = PortfolioSummary(
                positions={
                    "BTC-USD": Position(
                        symbol="BTC-USD",
                        quantity=Decimal("0.1"),
                        side="LONG",
                        leverage=1,
                        unrealized_pnl=Decimal("100.00"),
                        entry_price=Decimal("95000.00"),
                        mark_price=Decimal("95100.00"),
                    )
                }
            )

            # First call should update
            widget.on_state_updated(state)
            cached_sig = widget._last_display_signature
            assert cached_sig is not None

            # Second call with same state should hit cache
            widget.on_state_updated(state)
            assert widget._last_display_signature == cached_sig

    @pytest.mark.asyncio
    async def test_updates_when_pnl_changes(self) -> None:
        """Test that signature detects PnL changes."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.system_data = SystemStatus(connection_status="CONNECTED")
            state.position_data = PortfolioSummary(
                positions={
                    "BTC-USD": Position(
                        symbol="BTC-USD",
                        quantity=Decimal("0.1"),
                        side="LONG",
                        leverage=1,
                        unrealized_pnl=Decimal("100.00"),
                        entry_price=Decimal("95000.00"),
                        mark_price=Decimal("95100.00"),
                    )
                }
            )

            widget.on_state_updated(state)
            original_sig = widget._last_display_signature

            # Change PnL
            state.position_data = PortfolioSummary(
                positions={
                    "BTC-USD": Position(
                        symbol="BTC-USD",
                        quantity=Decimal("0.1"),
                        side="LONG",
                        leverage=1,
                        unrealized_pnl=Decimal("200.00"),  # Changed
                        entry_price=Decimal("95000.00"),
                        mark_price=Decimal("95200.00"),  # Changed
                    )
                }
            )

            widget.on_state_updated(state)
            assert widget._last_display_signature != original_sig

    @pytest.mark.asyncio
    async def test_signature_includes_connection_status(self) -> None:
        """Test that signature changes when connection status changes."""
        app = PositionCardTestApp()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)

            state = TuiState(validation_enabled=False, delta_updates_enabled=False)
            state.running = True
            state.system_data = SystemStatus(connection_status="CONNECTED")
            state.position_data = PortfolioSummary(positions={})

            widget.on_state_updated(state)
            sig_connected = widget._last_display_signature

            # Change connection status
            state.system_data = SystemStatus(connection_status="RECONNECTING")

            widget.on_state_updated(state)
            assert widget._last_display_signature != sig_connected


class TestPositionCardStateRegistry:
    """Tests for StateRegistry integration."""

    @pytest.mark.asyncio
    async def test_mount_registers_with_state_registry(self) -> None:
        """Test that widget registers with StateRegistry on mount."""

        class MockStateRegistry:
            def __init__(self):
                self.registered = []

            def register(self, widget):
                self.registered.append(widget)

            def unregister(self, widget):
                if widget in self.registered:
                    self.registered.remove(widget)

        class TestAppWithRegistry(App):
            def __init__(self):
                super().__init__()
                self.state_registry = MockStateRegistry()

            def compose(self) -> ComposeResult:
                yield PositionCardWidget(id="test-widget")

        app = TestAppWithRegistry()

        async with app.run_test():
            widget = app.query_one(PositionCardWidget)
            assert widget in app.state_registry.registered
