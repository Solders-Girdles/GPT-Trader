"""
Orders widget for displaying active trading orders.

This widget displays a table of active orders with symbol, side,
quantity, price, and status information.

Uses row keys for efficient DataTable updates - only adds/removes
changed rows instead of rebuilding the entire table.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.types import Order
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class OrdersWidget(Static):
    """Displays active orders in a data table.

    Shows pending and active orders with their details including
    symbol, side (BUY/SELL with color coding), quantity, price, and status.

    Uses row keys (order_id) for efficient DataTable updates.
    """

    # Styles moved to styles/widgets/portfolio.tcss

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        table = DataTable(id="orders-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield Label("", id="orders-empty", classes="empty-state")

    def on_mount(self) -> None:
        """Initialize the orders table with columns."""
        table = self.query_one("#orders-table", DataTable)
        # Add columns - alignment handled in add_row with Text objects
        table.add_columns("Symbol", "Side", "Quantity", "Price", "Status")

    @safe_update
    def update_orders(self, orders: list[Order]) -> None:
        """Update the orders table with current data.

        Uses row keys for efficient diffing - only adds/removes changed rows.

        Args:
            orders: List of Order objects to display.
        """
        table = self.query_one("#orders-table", DataTable)
        empty_label = self.query_one("#orders-empty", Label)

        # Show empty state or data
        if not orders:
            # Clear all rows when empty
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_label.display = True
            empty_label.update("No orders yet. Orders appear when the bot places trades.")
        else:
            table.display = True
            empty_label.display = False

            # Get current row keys
            existing_keys = set(table.rows.keys())
            new_keys = {order.order_id for order in orders}

            # Remove orders no longer present (filled/cancelled)
            for key in existing_keys - new_keys:
                try:
                    table.remove_row(key)
                    logger.debug(f"Removed order row: {key}")
                except Exception:
                    pass  # Row may not exist

            # Add/update orders
            for order in orders:
                row_data = self._format_order_row(order)

                if order.order_id in existing_keys:
                    # Update existing row in-place
                    try:
                        self._update_row_cells(table, order.order_id, row_data)
                    except Exception:
                        # Fallback: remove and re-add if update fails
                        try:
                            table.remove_row(order.order_id)
                        except Exception:
                            pass
                        table.add_row(*row_data, key=order.order_id)
                else:
                    # Add new order
                    table.add_row(*row_data, key=order.order_id)
                    logger.debug(f"Added new order row: {order.order_id}")

    def _format_order_row(self, order: Order) -> tuple:
        """Format an order into row data tuple.

        Args:
            order: Order object to format.

        Returns:
            Tuple of formatted cell values.
        """
        # Colorize Side
        side_color = THEME.colors.success if order.side == "BUY" else THEME.colors.error
        formatted_side = f"[{side_color}]{order.side}[/{side_color}]"

        return (
            order.symbol,
            formatted_side,  # Preserves color markup
            Text(format_quantity(order.quantity), justify="right"),
            Text(format_price(order.price), justify="right"),
            order.status,
        )

    def _update_row_cells(
        self,
        table: DataTable,
        row_key: str,
        row_data: tuple,
    ) -> None:
        """Update cells in an existing row.

        Args:
            table: DataTable to update.
            row_key: Key of the row to update.
            row_data: New data for the row cells.
        """
        # Get column keys
        columns = list(table.columns.keys())

        # Update each cell
        for col_key, value in zip(columns, row_data):
            table.update_cell(row_key, col_key, value)
