from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import (
    format_currency,
    format_percentage,
    format_price,
    format_quantity,
)
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.trade_matcher import TradeMatcher
from gpt_trader.tui.types import (
    Order,
    Position,
    Trade,
)
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class PositionsWidget(Static):
    """Displays active positions."""

    DEFAULT_CSS = """
    PositionsWidget {
        layout: vertical;
        height: 1fr;
    }

    PositionsWidget DataTable {
        height: 1fr;
    }
    """

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update positions automatically."""
        if state is None:
            return

        logger.debug(
            f"[PositionsWidget] State update: "
            f"positions={len(state.position_data.positions)}, "
            f"total_pnl={state.position_data.total_unrealized_pnl}, "
            f"timestamp={state.last_update_timestamp:.2f}"
        )

        self.update_positions(
            state.position_data.positions,
            state.position_data.total_unrealized_pnl,
            risk_data=state.risk_data.position_leverage,
        )

    def compose(self) -> ComposeResult:
        yield Label("💼 ACTIVE POSITIONS", classes="header")
        yield DataTable(id="positions-table", zebra_stripes=True)
        yield Label("", id="positions-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        # Add columns - alignment handled in add_row with Text objects
        table.add_columns("Symbol", "Quantity", "Entry", "Current", "PnL", "%", "Leverage")

    @safe_update(notify_user=True, error_tracker=True, severity="warning")
    def update_positions(
        self,
        positions: dict[str, Position],
        total_pnl: Decimal,
        risk_data: dict[str, float] | None = None,
    ) -> None:
        table = self.query_one(DataTable)
        empty_label = self.query_one("#positions-empty", Label)
        table.clear()

        # Show empty state or data
        if not positions:
            table.display = False
            empty_label.display = True

            # Mode-aware empty state message (single-line format)
            if self.app and hasattr(self.app, "data_source_mode"):
                mode = self.app.data_source_mode  # type: ignore[attr-defined]
                if mode == "read_only":
                    empty_label.update("📊 No positions • Read-only mode - observing market data")
                else:
                    empty_label.update("📊 No positions • Press [S] to start bot")
            else:
                empty_label.update("📊 No positions • Press [S] to start bot")
        else:
            table.display = True
            empty_label.display = False
            for symbol, pos in positions.items():
                # Calculate P&L percentage (now using Decimal directly)
                try:
                    if pos.entry_price > 0:
                        pnl_pct = ((pos.mark_price - pos.entry_price) / pos.entry_price) * 100
                        pnl_pct_str = format_percentage(Decimal(str(pnl_pct)))
                    else:
                        pnl_pct_str = "N/A"
                except (ValueError, ZeroDivisionError):
                    pnl_pct_str = "N/A"

                # Get leverage from risk data
                leverage_val = 1.0
                if risk_data and symbol in risk_data:
                    leverage_val = risk_data[symbol]

                # Format and color-code leverage
                leverage_str = f"{leverage_val:.1f}x"
                if leverage_val < 2.0:
                    leverage_display = f"[green]{leverage_str}[/green]"
                elif leverage_val < 5.0:
                    leverage_display = f"[yellow]{leverage_str}[/yellow]"
                else:
                    leverage_display = f"[red]{leverage_str}[/red]"

                # Use mark_price as current price, fallback to entry if not available
                current_price = (
                    pos.mark_price
                    if pos.mark_price and pos.mark_price != Decimal("0")
                    else pos.entry_price
                )

                # Right-align numeric columns using Text objects
                table.add_row(
                    pos.symbol,  # Left-aligned (symbol)
                    Text(format_quantity(pos.quantity), justify="right"),
                    Text(format_price(pos.entry_price), justify="right"),
                    Text(format_price(current_price), justify="right"),
                    Text(format_currency(pos.unrealized_pnl), justify="right"),
                    Text(pnl_pct_str, justify="right"),
                    Text.from_markup(leverage_display, justify="right"),  # Preserves markup
                )


class OrdersWidget(Static):
    """Widget to display active orders."""

    DEFAULT_CSS = """
    OrdersWidget {
        layout: vertical;
        height: 1fr;
    }

    OrdersWidget DataTable {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield DataTable(id="orders-table")
        yield Label("", id="orders-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one("#orders-table", DataTable)
        # Add columns - alignment handled in add_row with Text objects
        table.add_columns("Symbol", "Side", "Quantity", "Price", "Status")

    @safe_update
    def update_orders(self, orders: list[Order]) -> None:
        table = self.query_one("#orders-table", DataTable)
        empty_label = self.query_one("#orders-empty", Label)
        table.clear()

        if not orders:
            table.display = False
            empty_label.display = True
            empty_label.update("📋 No orders • Orders appear when bot places trades")
        else:
            table.display = True
            empty_label.display = False
            for order in orders:
                # Colorize Side
                side_color = THEME.colors.success if order.side == "BUY" else THEME.colors.error
                formatted_side = f"[{side_color}]{order.side}[/{side_color}]"

                # Right-align numeric columns using Text objects
                table.add_row(
                    order.symbol,
                    formatted_side,  # Preserves color markup
                    Text(format_quantity(order.quantity), justify="right"),
                    Text(format_price(order.price), justify="right"),
                    order.status,
                )


class TradesWidget(Static):
    """Widget to display recent trades."""

    DEFAULT_CSS = """
    TradesWidget {
        layout: vertical;
        height: 1fr;
    }

    TradesWidget DataTable {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield DataTable(id="trades-table")
        yield Label("", id="trades-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one("#trades-table", DataTable)
        # Add columns - alignment handled in add_row with Text objects
        table.add_columns("Symbol", "Side", "Quantity", "Price", "Order ID", "P&L", "Time")
        # Initialize trade matcher
        self._trade_matcher = TradeMatcher()

    @safe_update
    def update_trades(self, trades: list[Trade]) -> None:
        table = self.query_one("#trades-table", DataTable)
        empty_label = self.query_one("#trades-empty", Label)
        table.clear()

        if not trades:
            table.display = False
            empty_label.display = True
            empty_label.update("📈 No trades • Trade history appears after execution")
        else:
            table.display = True
            empty_label.display = False

            # Process trades to calculate P&L matches
            pnl_map = self._trade_matcher.process_trades(trades)

            for trade in trades:
                # Colorize Side
                side_color = THEME.colors.success if trade.side == "BUY" else THEME.colors.error
                formatted_side = f"[{side_color}]{trade.side}[/{side_color}]"

                # Simplify time string if needed (already done in state.py or kept raw)
                # But let's keep it simple here as it's already a string in Trade object
                time_str = trade.time
                if "T" in time_str:
                    try:
                        time_str = time_str.split("T")[1].split(".")[0]
                    except IndexError:
                        pass

                # Get P&L display value and apply color coding
                pnl_str = pnl_map.get(trade.trade_id, "N/A")

                if pnl_str == "N/A":
                    pnl_display = Text(pnl_str, style="dim", justify="right")
                else:
                    try:
                        pnl_value = float(pnl_str)
                        if pnl_value > 0:
                            pnl_markup = (
                                f"[{THEME.colors.success}]{pnl_str}[/{THEME.colors.success}]"
                            )
                            pnl_display = Text.from_markup(pnl_markup, justify="right")
                        elif pnl_value < 0:
                            pnl_markup = f"[{THEME.colors.error}]{pnl_str}[/{THEME.colors.error}]"
                            pnl_display = Text.from_markup(pnl_markup, justify="right")
                        else:
                            # Zero P&L - neutral white
                            pnl_display = Text(pnl_str, justify="right")
                    except ValueError:
                        # Fallback if parsing fails
                        pnl_display = Text(pnl_str, style="dim", justify="right")

                # Right-align numeric columns using Text objects
                table.add_row(
                    trade.symbol,
                    formatted_side,  # Preserves color markup
                    Text(format_quantity(trade.quantity), justify="right"),
                    Text(format_price(trade.price), justify="right"),
                    trade.order_id[-8:] if trade.order_id else "",
                    pnl_display,  # NEW: P&L column
                    Text(time_str, justify="right"),
                )
