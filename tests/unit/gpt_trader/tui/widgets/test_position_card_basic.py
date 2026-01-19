"""Tests for PositionCardWidget basic behavior."""

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
