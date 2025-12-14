"""
Market overlay screen for real-time market data.

This screen provides access to market data moved from the main view:
- Real-time prices for tracked symbols
- Sparkline trend indicators
- Price change percentages
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import Screen
from gpt_trader.tui.widgets import ContextualFooter
from gpt_trader.tui.widgets.shell import CommandBar

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import MarketWatchWidget
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class MarketScreen(Screen):
    """Market overlay showing real-time price data.

    Displays comprehensive market information:
    - Symbol prices with color-coded changes
    - Sparkline trend indicators
    - Last update timestamps

    Keyboard navigation:
    - ESC/Q/M: Close and return to main screen
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("m", "dismiss", "Close"),  # Toggle behavior
    ]

    # Reactive state property
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update market data."""
        if state is None:
            return

        try:
            market_widget = self.query_one("#market-full", MarketWatchWidget)
            market_widget.update_market(state.market_data)
        except Exception as e:
            logger.debug(f"Failed to update MarketWatchWidget: {e}")

    def compose(self) -> ComposeResult:
        """Compose the market screen layout."""
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )

        with Container(id="market-container"):
            yield MarketWatchWidget(id="market-full", classes="dashboard-item")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize with current state when mounted."""
        logger.debug("MarketScreen mounted")
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)
        if hasattr(self.app, "tui_state"):
            self.state = self.app.tui_state  # type: ignore[attr-defined]

    def on_unmount(self) -> None:
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        self.state = state

    def action_dismiss(self, result: object = None) -> None:
        """Close the market screen and return to main view."""
        self.app.pop_screen()
