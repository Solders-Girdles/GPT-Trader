"""Tests for PositionCardWidget signature caching."""

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

            widget.on_state_updated(state)
            cached_sig = widget._last_display_signature
            assert cached_sig is not None

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

            state.position_data = PortfolioSummary(
                positions={
                    "BTC-USD": Position(
                        symbol="BTC-USD",
                        quantity=Decimal("0.1"),
                        side="LONG",
                        leverage=1,
                        unrealized_pnl=Decimal("200.00"),
                        entry_price=Decimal("95000.00"),
                        mark_price=Decimal("95200.00"),
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

            state.system_data = SystemStatus(connection_status="RECONNECTING")

            widget.on_state_updated(state)
            assert widget._last_display_signature != sig_connected
