"""
System details and diagnostics screen.

This screen provides detailed system health information and account
details for monitoring and debugging.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Label

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import AccountWidget, ContextualFooter
from gpt_trader.tui.widgets.shell import CommandBar
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class SystemDetailsScreen(Screen):
    """Detailed system health and diagnostics screen.

    Displays comprehensive system information including:
    - Connection status and API health
    - Rate limiting information
    - Account details and balances
    - Portfolio value and P&L summary
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    # Reactive state property
    state = reactive(None)  # Type: TuiState | None

    @property
    def observer_priority(self) -> int:
        """High priority so screen updates before individual widgets."""
        return 100

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update system details widgets."""
        if state is None:
            return

        # Update SystemHealthWidget
        try:
            from gpt_trader.tui.widgets.system import SystemHealthWidget

            system_widget = self.query_one(SystemHealthWidget)
            if hasattr(system_widget, "update_system"):
                system_widget.update_system(state.system_data)
            if hasattr(system_widget, "update_websocket"):
                system_widget.update_websocket(state.websocket_data)
            if hasattr(system_widget, "update_metrics"):
                system_widget.update_metrics(state.metrics_data)
        except Exception as e:
            logger.debug(f"Failed to update SystemHealthWidget: {e}")

        # Update AccountWidget
        try:
            account_widget = self.query_one(AccountWidget)
            if hasattr(account_widget, "update_account"):
                account_widget.update_account(
                    state.account_data,
                    portfolio_value=state.position_data.equity,
                    unrealized_pnl=state.position_data.total_unrealized_pnl,
                )
        except Exception as e:
            logger.debug(f"Failed to update AccountWidget: {e}")

    def compose(self) -> ComposeResult:
        """Compose the system details screen layout."""
        from gpt_trader.tui.widgets.system import SystemHealthWidget

        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )
        yield Label("SYSTEM DETAILS", classes="screen-header")
        with Container(id="system-details-container"):
            yield SystemHealthWidget(
                id="detailed-system", compact_mode=False, classes="dashboard-item"
            )
            yield AccountWidget(id="detailed-account", compact_mode=False, classes="dashboard-item")
        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize with current state when mounted."""
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
        """Close the system details screen and return to main view."""
        self.app.pop_screen()
