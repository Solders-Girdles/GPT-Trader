"""Unified portfolio widget combining Positions, Orders, and Trades."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Label, Static, TabbedContent, TabPane

from gpt_trader.tui.widgets.portfolio import OrdersWidget, PositionsWidget, TradesWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class PortfolioWidget(Static):
    """Unified portfolio view with Positions, Orders, and Trades tabs.

    This widget provides a tabbed interface to view:
    - Active positions with P&L
    - Pending orders
    - Trade history with matched P&L

    Keyboard shortcuts:
    - j/k: Navigate between tabs
    - 1/2/3: Jump to specific tab
    """

    BINDINGS = [
        ("right", "next_tab", "Next Tab"),
        ("left", "previous_tab", "Previous Tab"),
        ("j", "next_tab", "Next Tab"),  # Vim-style for power users
        ("k", "previous_tab", "Previous Tab"),
        ("1", "positions_tab", "Positions"),
        ("2", "orders_tab", "Orders"),
        ("3", "trades_tab", "Trades"),
    ]

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update portfolio automatically."""
        if state is None:
            return

        logger.debug(
            f"[PortfolioWidget] State update: "
            f"positions={len(state.position_data.positions)}, "
            f"orders={len(state.order_data.orders)}, "
            f"trades={len(state.trade_data.trades)}"
        )

        # Update all child widgets with error isolation
        try:
            positions_widget = self.query_one(PositionsWidget)
            positions_widget.update_positions(
                state.position_data.positions,
                state.position_data.total_unrealized_pnl,
            )
        except Exception as e:
            logger.error(f"Failed to update positions in portfolio: {e}")

        try:
            orders_widget = self.query_one(OrdersWidget)
            orders_widget.update_orders(
                state.order_data.orders,
                trades=state.trade_data.trades,
            )
        except Exception as e:
            logger.error(f"Failed to update orders in portfolio: {e}")

        try:
            trades_widget = self.query_one(TradesWidget)
            trades_widget.update_trades(state.trade_data.trades)
        except Exception as e:
            logger.error(f"Failed to update trades in portfolio: {e}")

    def action_next_tab(self) -> None:
        """Move to next tab."""
        tabs = self.query_one(TabbedContent)
        tabs.action_next_tab()

    def action_previous_tab(self) -> None:
        """Move to previous tab."""
        tabs = self.query_one(TabbedContent)
        tabs.action_previous_tab()

    def action_positions_tab(self) -> None:
        """Jump to Positions tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "positions-tab"

    def action_orders_tab(self) -> None:
        """Jump to Orders tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "orders-tab"

    def action_trades_tab(self) -> None:
        """Jump to Trades tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "trades-tab"

    def on_mount(self) -> None:
        """Register with state registry on mount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from state registry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Called by StateRegistry when state changes."""
        self.state = state

    def compose(self) -> ComposeResult:
        """Compose the portfolio widget layout."""
        yield Label("PORTFOLIO", classes="widget-header")
        with TabbedContent():
            with TabPane("Positions", id="positions-tab"):
                yield PositionsWidget(id="portfolio-positions")
            with TabPane("Orders", id="orders-tab"):
                yield OrdersWidget(id="portfolio-orders")
            with TabPane("Trades", id="trades-tab"):
                yield TradesWidget(id="portfolio-trades")
