"""Unified portfolio widget combining Positions, Orders, and Trades."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import Label, Static, TabbedContent, TabPane

from gpt_trader.tui.widgets.positions import OrdersWidget, PositionsWidget, TradesWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class PortfolioWidget(Static):
    """Unified portfolio view with Positions, Orders, and Trades tabs."""

    DEFAULT_CSS = """
    PortfolioWidget {
        layout: vertical;
        height: 1fr;
        border-top: solid $border-subtle;
        background: $bg-secondary;
        padding: 0;
    }

    PortfolioWidget TabbedContent {
        height: 1fr;
        border: none;
    }

    PortfolioWidget TabPane {
        padding: 0;
    }
    """

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
                risk_data=state.risk_data.position_leverage,
            )
        except Exception as e:
            logger.error(f"Failed to update positions in portfolio: {e}")

        try:
            orders_widget = self.query_one(OrdersWidget)
            orders_widget.update_orders(state.order_data.orders)
        except Exception as e:
            logger.error(f"Failed to update orders in portfolio: {e}")

        try:
            trades_widget = self.query_one(TradesWidget)
            trades_widget.update_trades(state.trade_data.trades)
        except Exception as e:
            logger.error(f"Failed to update trades in portfolio: {e}")

    def compose(self) -> ComposeResult:
        yield Label("💼 PORTFOLIO", classes="header")
        with TabbedContent():
            with TabPane("📊 Positions", id="positions-tab"):
                yield PositionsWidget(id="portfolio-positions")
            with TabPane("📋 Orders", id="orders-tab"):
                yield OrdersWidget(id="portfolio-orders")
            with TabPane("📈 Trades", id="trades-tab"):
                yield TradesWidget(id="portfolio-trades")
