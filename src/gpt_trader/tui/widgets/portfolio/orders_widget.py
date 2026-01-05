"""
Orders widget for displaying active trading orders.

This widget displays a table of active orders with symbol, side,
quantity, price, fill progress, average fill price, age, and status.

Uses row keys for efficient DataTable updates - only adds/removes
changed rows instead of rebuilding the entire table.

Supports trade-linking: when order.filled_quantity/avg_fill_price are
missing, fill info can be derived from matching trades.
"""

from __future__ import annotations

from decimal import Decimal

from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import DataTable, Static

from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.staleness_helpers import format_freshness_label
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.types import Order, Trade
from gpt_trader.tui.utilities import get_age_seconds
from gpt_trader.tui.widgets.table_copy_mixin import TableCopyMixin
from gpt_trader.tui.widgets.tile_states import TileEmptyState
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")

# Order age thresholds for status coloring (seconds)
ORDER_AGE_WARNING_SECONDS = 30
ORDER_AGE_CRITICAL_SECONDS = 60


def _build_order_fill_index(
    trades: list[Trade],
) -> dict[str, tuple[Decimal, Decimal]]:
    """Build an index of fill info per order from trade data.

    For each order_id, computes total filled quantity and volume-weighted
    average fill price from matching trades.

    Args:
        trades: List of Trade objects.

    Returns:
        Dict mapping order_id to (total_filled_qty, weighted_avg_price).
    """
    # Aggregate: order_id -> (total_qty, total_notional)
    aggregates: dict[str, tuple[Decimal, Decimal]] = {}

    for trade in trades:
        if not trade.order_id:
            continue

        order_id = trade.order_id
        trade_qty = trade.quantity
        trade_price = trade.price
        notional = trade_qty * trade_price

        if order_id in aggregates:
            prev_qty, prev_notional = aggregates[order_id]
            aggregates[order_id] = (prev_qty + trade_qty, prev_notional + notional)
        else:
            aggregates[order_id] = (trade_qty, notional)

    # Convert to (total_qty, avg_price)
    result: dict[str, tuple[Decimal, Decimal]] = {}
    for order_id, (total_qty, total_notional) in aggregates.items():
        if total_qty > 0:
            avg_price = total_notional / total_qty
            result[order_id] = (total_qty, avg_price)

    return result


class OrdersWidget(TableCopyMixin, Static):
    """Displays active orders in a data table.

    Shows pending and active orders with their details including
    symbol, side (BUY/SELL with color coding), quantity, price, and status.

    Uses row keys (order_id) for efficient DataTable updates.

    Keyboard shortcuts:
        c: Copy selected row to clipboard
        C: Copy all rows to clipboard
    """

    BINDINGS = [
        *TableCopyMixin.COPY_BINDINGS,
    ]

    # Styles moved to styles/widgets/portfolio.tcss

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        table = DataTable(id="orders-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield TileEmptyState(
            title="No Active Orders",
            subtitle="Orders appear when the bot places trades",
            icon="◌",
            actions=["[S] Start Bot", "[R] Refresh"],
            id="orders-empty",
        )

    def on_mount(self) -> None:
        """Initialize the orders table with columns."""
        table = self.query_one("#orders-table", DataTable)
        # Add columns - short headers for width; alignment in add_row
        # Fill%: fill progress, Avg Px: average fill price, Age: time since creation
        table.add_columns("Symbol", "Side", "Qty", "Price", "Fill%", "Avg Px", "Age", "Status")

    @safe_update
    def update_orders(
        self,
        orders: list[Order],
        trades: list[Trade] | None = None,
    ) -> None:
        """Update the orders table with current data.

        Uses row keys for efficient diffing - only adds/removes changed rows.
        When trades are provided, can derive fill info for orders missing
        filled_quantity/avg_fill_price.

        Args:
            orders: List of Order objects to display.
            trades: Optional list of Trade objects to derive fill info from.
        """
        table = self.query_one("#orders-table", DataTable)
        empty_state = self.query_one("#orders-empty", TileEmptyState)

        # Build fill index from trades if provided
        fill_index = _build_order_fill_index(trades) if trades else {}

        # Show empty state or data
        if not orders:
            # Clear all rows when empty
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_state.display = True
            # Keep default message - orders are only for active trading
            return
        else:
            table.display = True
            empty_state.display = False

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
                row_data = self._format_order_row(order, fill_index)

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

    def _format_order_row(
        self,
        order: Order,
        fill_index: dict[str, tuple[Decimal, Decimal]],
    ) -> tuple:
        """Format an order into row data tuple.

        Args:
            order: Order object to format.
            fill_index: Dict mapping order_id to (filled_qty, avg_price) from trades.

        Returns:
            Tuple of formatted cell values for all columns.
        """
        # Colorize Side
        side_color = THEME.colors.success if order.side == "BUY" else THEME.colors.error
        formatted_side = f"[{side_color}]{order.side}[/{side_color}]"

        # Get fill info: prefer order fields, fall back to trade-derived
        filled_qty, avg_fill_price = self._get_fill_info(order, fill_index)

        # Calculate fill percentage
        filled_pct = self._format_fill_progress(order.quantity, filled_qty)

        # Format average fill price
        avg_px_display = self._format_avg_fill_price(avg_fill_price)

        # Calculate and format order age
        age_display = self._format_order_age(order)

        return (
            order.symbol,
            formatted_side,  # Preserves color markup
            Text(format_quantity(order.quantity), justify="right"),
            Text(format_price(order.price), justify="right"),
            Text(filled_pct, justify="right"),
            avg_px_display,
            age_display,
            order.status,
        )

    def _get_fill_info(
        self,
        order: Order,
        fill_index: dict[str, tuple[Decimal, Decimal]],
    ) -> tuple[Decimal, Decimal | None]:
        """Get fill info for an order, preferring order fields over trade-derived.

        Args:
            order: Order object.
            fill_index: Dict mapping order_id to (filled_qty, avg_price) from trades.

        Returns:
            Tuple of (filled_quantity, avg_fill_price or None).
        """
        # Prefer order's own fill data if present
        if order.filled_quantity > 0:
            return order.filled_quantity, order.avg_fill_price

        # Fall back to trade-derived data
        if order.order_id in fill_index:
            return fill_index[order.order_id]

        # No fill info available
        return Decimal("0"), None

    def _format_fill_progress(self, total_qty: Decimal, filled_qty: Decimal) -> str:
        """Format fill progress as percentage.

        Args:
            total_qty: Total order quantity.
            filled_qty: Filled quantity.

        Returns:
            Fill percentage string (e.g., "75%", "0%", "100%").
        """
        if total_qty <= 0:
            return "0%"

        filled = float(filled_qty)
        total = float(total_qty)

        if total <= 0:
            return "0%"

        pct = (filled / total) * 100
        return f"{pct:.0f}%"

    def _format_avg_fill_price(self, avg_price: Decimal | None) -> Text:
        """Format average fill price for display.

        Args:
            avg_price: Average fill price or None if no fills.

        Returns:
            Rich Text with formatted price or "--" if no fills.
        """
        if avg_price is None or avg_price <= 0:
            return Text("--", justify="right", style=THEME.colors.text_muted)

        return Text(format_price(avg_price), justify="right")

    def _format_order_age(self, order: Order) -> Text:
        """Format order age with color coding.

        Args:
            order: Order object.

        Returns:
            Rich Text with colored age display.
        """
        age = get_age_seconds(order.creation_time)

        if age is None:
            return Text("--", justify="right")

        # Format as relative time
        age_str = format_freshness_label(age)

        # Color based on age thresholds
        if age >= ORDER_AGE_CRITICAL_SECONDS:
            color = THEME.colors.error
        elif age >= ORDER_AGE_WARNING_SECONDS:
            color = THEME.colors.warning
        else:
            color = THEME.colors.text_muted

        return Text(age_str, style=color, justify="right")

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
