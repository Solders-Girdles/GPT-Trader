from textual.app import ComposeResult
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.types import (
    Order,
    Position,
    Trade,
)


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

    def compose(self) -> ComposeResult:
        yield Label("💼 ACTIVE POSITIONS", classes="header")
        yield DataTable(id="positions-table", zebra_stripes=True)
        yield Label("", id="positions-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Symbol", "Quantity", "Entry", "Current", "PnL", "%", "Leverage")

    @safe_update
    def update_positions(
        self,
        positions: dict[str, Position],
        total_pnl: str,
        risk_data: dict[str, float] | None = None,
    ) -> None:
        table = self.query_one(DataTable)
        empty_label = self.query_one("#positions-empty", Label)
        table.clear()

        # Show empty state or data
        if not positions:
            table.display = False
            empty_label.display = True
            empty_label.update(
                "📊 No open positions\n\n"
                "💡 Start the bot to begin trading\n"
                "Press [S] to start"
            )
        else:
            table.display = True
            empty_label.display = False
            for symbol, pos in positions.items():
                # Calculate P&L percentage
                try:
                    entry = float(pos.entry_price.replace("$", "").replace(",", ""))
                    current = float(pos.mark_price.replace("$", "").replace(",", ""))
                    if entry > 0:
                        pnl_pct = ((current - entry) / entry) * 100
                        pnl_pct_str = f"{pnl_pct:+.2f}%"
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
                    if pos.mark_price and pos.mark_price != "0.00"
                    else pos.entry_price
                )

                table.add_row(
                    pos.symbol,
                    pos.quantity,
                    pos.entry_price,
                    current_price,
                    pos.unrealized_pnl,
                    pnl_pct_str,
                    leverage_display,
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
        yield Label("📋 ACTIVE ORDERS", classes="header")
        yield DataTable(id="orders-table")
        yield Label("", id="orders-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one("#orders-table", DataTable)
        table.add_columns("Symbol", "Side", "Quantity", "Price", "Status")

    @safe_update
    def update_orders(self, orders: list[Order]) -> None:
        table = self.query_one("#orders-table", DataTable)
        empty_label = self.query_one("#orders-empty", Label)
        table.clear()

        if not orders:
            table.display = False
            empty_label.display = True
            empty_label.update(
                "📋 No active orders\n\n" "💡 Orders will appear here when the bot places trades"
            )
        else:
            table.display = True
            empty_label.display = False
            for order in orders:
                # Colorize Side
                side_color = THEME.colors.success if order.side == "BUY" else THEME.colors.error
                formatted_side = f"[{side_color}]{order.side}[/{side_color}]"

                table.add_row(
                    order.symbol, formatted_side, order.quantity, order.price, order.status
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
        yield Label("📈 RECENT TRADES", classes="header")
        yield DataTable(id="trades-table")
        yield Label("", id="trades-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one("#trades-table", DataTable)
        table.add_columns("Symbol", "Side", "Quantity", "Price", "Order ID", "Time")

    @safe_update
    def update_trades(self, trades: list[Trade]) -> None:
        table = self.query_one("#trades-table", DataTable)
        empty_label = self.query_one("#trades-empty", Label)
        table.clear()

        if not trades:
            table.display = False
            empty_label.display = True
            empty_label.update(
                "📈 No recent trades\n\n" "💡 Trade history will appear here after execution"
            )
        else:
            table.display = True
            empty_label.display = False
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

                table.add_row(
                    trade.symbol,
                    formatted_side,
                    trade.quantity,
                    trade.price,
                    trade.order_id,
                    time_str,
                )
