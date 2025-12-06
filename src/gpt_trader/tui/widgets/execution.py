"""Tabbed execution widget combining Orders and Trades."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Label, Static, TabbedContent, TabPane

from gpt_trader.tui.widgets.portfolio import OrdersWidget, TradesWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class ExecutionWidget(Static):
    """Tabbed widget showing Orders and Trades."""

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update execution automatically."""
        if state is None:
            return

        logger.debug(
            f"[ExecutionWidget] State update: "
            f"orders={len(state.order_data.orders)}, "
            f"trades={len(state.trade_data.trades)}"
        )

        self.update_orders(state.order_data.orders)
        self.update_trades(state.trade_data.trades)

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
        yield Label("📊 EXECUTION", classes="header")
        with TabbedContent():
            with TabPane("📋 Orders", id="orders-tab"):
                yield OrdersWidget(id="orders-content")
            with TabPane("📈 Trades", id="trades-tab"):
                yield TradesWidget(id="trades-content")

    def update_orders(self, orders: list) -> None:
        """Update orders table."""
        self.query_one(OrdersWidget).update_orders(orders)

    def update_trades(self, trades: list) -> None:
        """Update trades table."""
        self.query_one(TradesWidget).update_trades(trades)
