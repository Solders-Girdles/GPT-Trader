from textual.app import ComposeResult
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import (
    Order,
    Position,
    Trade,
)


class PositionsWidget(Static):
    """Displays active positions."""

    def compose(self) -> ComposeResult:
        yield Label("ACTIVE POSITIONS", classes="header")
        yield DataTable(id="positions-table", zebra_stripes=True)
        yield Label("", id="positions-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Sym", "Qty", "Entry", "PnL")

    @safe_update
    def update_positions(self, positions: dict[str, Position], total_pnl: str) -> None:
        table = self.query_one(DataTable)
        empty_label = self.query_one("#positions-empty", Label)
        table.clear()

        # Show empty state or data
        if not positions:
            table.display = False
            empty_label.display = True
            empty_label.update("No open positions")
        else:
            table.display = True
            empty_label.display = False
            for symbol, pos in positions.items():
                table.add_row(pos.symbol, pos.quantity, pos.entry_price, pos.unrealized_pnl)


class OrdersWidget(Static):
    """Widget to display active orders."""

    def compose(self) -> ComposeResult:
        yield Label("Active Orders", classes="header")
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
            empty_label.update("No active orders")
        else:
            table.display = True
            empty_label.display = False
            for order in orders:
                # Colorize Side
                side_style = "green" if order.side == "BUY" else "red"
                formatted_side = f"[{side_style}]{order.side}[/{side_style}]"

                table.add_row(
                    order.symbol, formatted_side, order.quantity, order.price, order.status
                )


class TradesWidget(Static):
    """Widget to display recent trades."""

    def compose(self) -> ComposeResult:
        yield Label("Recent Trades", classes="header")
        yield DataTable(id="trades-table")
        yield Label("", id="trades-empty", classes="empty-state")

    def on_mount(self) -> None:
        table = self.query_one("#trades-table", DataTable)
        table.add_columns("Symbol", "Side", "Qty", "Price", "Order ID", "Time")

    @safe_update
    def update_trades(self, trades: list[Trade]) -> None:
        table = self.query_one("#trades-table", DataTable)
        empty_label = self.query_one("#trades-empty", Label)
        table.clear()

        if not trades:
            table.display = False
            empty_label.display = True
            empty_label.update("No recent trades")
        else:
            table.display = True
            empty_label.display = False
            for trade in trades:
                # Colorize Side
                side_style = "green" if trade.side == "BUY" else "red"
                formatted_side = f"[{side_style}]{trade.side}[/{side_style}]"

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
