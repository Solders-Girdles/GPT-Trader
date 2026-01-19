"""Tests for PositionCardWidget resilience with partial state."""

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

            widget.on_state_updated(state)

            assert not widget._bot_running
