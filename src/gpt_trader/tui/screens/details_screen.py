"""
Details overlay screen for portfolio, risk, and account information.

This screen provides access to secondary information moved from the main view:
- Positions (current holdings with P&L)
- Orders (pending orders)
- Trades (execution history)
- Risk (daily loss limits, leverage, guards)
- Account (balances, fees)
- System (API health, connection status)
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import TabbedContent, TabPane

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    AccountWidget,
    ContextualFooter,
    OrdersWidget,
    PositionsWidget,
    RiskWidget,
    SystemHealthWidget,
    TradesWidget,
)
from gpt_trader.tui.widgets.shell import CommandBar
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class DetailsScreen(Screen):
    """Details overlay with tabbed access to portfolio, risk, and account info.

    Provides comprehensive access to:
    - Positions: Current holdings with P&L tracking
    - Orders: Pending and active orders
    - Trades: Execution history with matched P&L
    - Risk: Daily loss limits, leverage, active guards
    - Account: Balances, fees, portfolio value
    - System: API health, connection status, rate limits

    Keyboard navigation:
    - 1-6: Switch tabs directly
    - Tab/Shift+Tab: Cycle through tabs
    - ESC/Q: Close and return to main screen
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
        ("1", "switch_tab('positions')", "Positions"),
        ("2", "switch_tab('orders')", "Orders"),
        ("3", "switch_tab('trades')", "Trades"),
        ("4", "switch_tab('risk')", "Risk"),
        ("5", "switch_tab('account')", "Account"),
        ("6", "switch_tab('system')", "System"),
    ]

    # Reactive state property
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update all detail widgets."""
        if state is None:
            return

        # Update PositionsWidget
        try:
            positions_widget = self.query_one("#details-positions", PositionsWidget)
            positions_widget.update_positions(
                state.position_data.positions,
                state.position_data.total_unrealized_pnl,
            )
        except Exception as e:
            logger.debug(f"Failed to update PositionsWidget: {e}")

        # Update OrdersWidget with trade data for fill derivation
        try:
            orders_widget = self.query_one("#details-orders", OrdersWidget)
            orders_widget.update_orders(
                state.order_data.orders,
                trades=state.trade_data.trades,
            )
        except Exception as e:
            logger.debug(f"Failed to update OrdersWidget: {e}")

        # Update TradesWidget
        try:
            trades_widget = self.query_one("#details-trades", TradesWidget)
            trades_widget.update_trades(state.trade_data.trades, state)
        except Exception as e:
            logger.debug(f"Failed to update TradesWidget: {e}")

        # Update RiskWidget
        try:
            risk_widget = self.query_one("#details-risk", RiskWidget)
            if hasattr(risk_widget, "update_risk"):
                risk_widget.update_risk(state.risk_data)
        except Exception as e:
            logger.debug(f"Failed to update RiskWidget: {e}")

        # Update AccountWidget
        try:
            account_widget = self.query_one("#details-account", AccountWidget)
            if hasattr(account_widget, "update_account"):
                account_widget.update_account(
                    state.account_data,
                    portfolio_value=state.position_data.equity,
                    unrealized_pnl=state.position_data.total_unrealized_pnl,
                )
        except Exception as e:
            logger.debug(f"Failed to update AccountWidget: {e}")

        # Update SystemHealthWidget
        try:
            system_widget = self.query_one("#details-system", SystemHealthWidget)
            if hasattr(system_widget, "update_system"):
                system_widget.update_system(state.system_data)
            if hasattr(system_widget, "update_websocket"):
                system_widget.update_websocket(state.websocket_data)
            if hasattr(system_widget, "update_metrics"):
                system_widget.update_metrics(state.metrics_data)
        except Exception as e:
            logger.debug(f"Failed to update SystemHealthWidget: {e}")

    def compose(self) -> ComposeResult:
        """Compose the details screen with tabbed layout."""
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )

        with Container(id="details-container"):
            with TabbedContent(id="details-tabs"):
                with TabPane("Positions", id="positions-tab"):
                    yield PositionsWidget(id="details-positions")

                with TabPane("Orders", id="orders-tab"):
                    yield OrdersWidget(id="details-orders")

                with TabPane("Trades", id="trades-tab"):
                    yield TradesWidget(id="details-trades")

                with TabPane("Risk", id="risk-tab"):
                    yield RiskWidget(id="details-risk")

                with TabPane("Account", id="account-tab"):
                    yield AccountWidget(id="details-account", compact_mode=False)

                with TabPane("System", id="system-tab"):
                    yield SystemHealthWidget(id="details-system", compact_mode=False)

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize with current state when mounted."""
        logger.debug("DetailsScreen mounted")
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
        """Close the details screen and return to main view."""
        self.app.pop_screen()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to a specific tab by ID."""
        try:
            tabs = self.query_one("#details-tabs", TabbedContent)
            tabs.active = f"{tab_id}-tab"
        except Exception as e:
            logger.debug(f"Failed to switch to tab {tab_id}: {e}")
