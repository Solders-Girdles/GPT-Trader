"""
Orders widget for displaying active trading orders.

This widget displays a table of active orders with symbol, side,
quantity, price, fill progress, average fill price, age, and status.

Uses row keys for efficient DataTable updates - only adds/removes
changed rows instead of rebuilding the entire table.

Supports trade-linking: when order.filled_quantity/avg_fill_price are
missing, fill info can be derived from matching trades.
"""

# naming: allow - qty is standard trading abbreviation for quantity

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import format_price, format_quantity
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.staleness_helpers import format_freshness_label
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.thresholds import (
    StatusLevel,
    get_order_age_status,
    get_order_status_level,
    get_status_color,
)
from gpt_trader.tui.types import Order, Trade
from gpt_trader.tui.utilities import get_age_seconds, get_sort_indicator, sort_table_data
from gpt_trader.tui.widgets.table_copy_mixin import TableCopyMixin
from gpt_trader.tui.widgets.tile_states import TileEmptyState
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


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
    """Displays active orders in a data table with sorting.

    Shows pending and active orders with their details including
    symbol, side (BUY/SELL with color coding), quantity, price, and status.

    Uses row keys (order_id) for efficient DataTable updates.

    Keyboard shortcuts:
        c: Copy selected row to clipboard
        C: Copy all rows to clipboard
        s: Cycle sort column (Fill%, Age, none)
        Enter: Open order detail modal
    """

    BINDINGS = [
        *TableCopyMixin.COPY_BINDINGS,
        Binding("s", "cycle_sort", "Sort", show=True),
        Binding("enter", "show_order_detail", "Details", show=True),
    ]

    # Sortable columns: fill_pct (Fill%), age
    _sortable_columns = ["fill_pct", "age"]

    # Sort state - reactive properties trigger watchers on change
    sort_column: reactive[str | None] = reactive(None)
    sort_ascending: reactive[bool] = reactive(True)

    # Cached data for re-sorting
    _orders_data: list[dict[str, Any]]
    _fill_index: dict[str, tuple[Decimal, Decimal]]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize OrdersWidget with sort state."""
        super().__init__(**kwargs)
        self._orders_data = []
        self._fill_index = {}
        self._trades: list[Trade] = []
        # Force reactive property initialization to prevent watcher triggering
        # during first access in _refresh_table
        _ = self.sort_column
        _ = self.sort_ascending

    def compose(self) -> ComposeResult:
        """Compose the widget layout."""
        yield Label("", id="orders-sort-indicator", classes="sort-hint")
        table: DataTable[str] = DataTable(id="orders-table", zebra_stripes=True)
        table.can_focus = True
        table.cursor_type = "row"
        yield table
        yield TileEmptyState(
            title="No Active Orders",
            subtitle="Orders appear when the bot places trades",
            icon="◌",
            actions=["[S] Sort", "[R] Refresh"],
            id="orders-empty",
        )

    def on_mount(self) -> None:
        """Initialize the orders table with columns."""
        table = self.query_one("#orders-table", DataTable)
        # Add columns - short headers for width; alignment in add_row
        # Fill%: fill progress, Avg Px: average fill price, Age: time since creation
        table.add_columns("Symbol", "Side", "Qty", "Price", "Fill%", "Avg Px", "Age", "Status")
        # Initialize sort indicator
        self._update_sort_indicator()

    def watch_sort_column(self, column: str | None) -> None:
        """Handle sort column change - refresh table display."""
        self._update_sort_indicator()
        # Only refresh if we have data (avoids re-render during init)
        if self._orders_data:
            self._refresh_table()

    def watch_sort_ascending(self, ascending: bool) -> None:
        """Handle sort direction change - refresh table display."""
        self._update_sort_indicator()
        # Only refresh if we have data (avoids re-render during init)
        if self._orders_data:
            self._refresh_table()

    def _update_sort_indicator(self) -> None:
        """Update the sort indicator label with sort state and Enter hint."""
        try:
            indicator = self.query_one("#orders-sort-indicator", Label)
            if self.sort_column is None:
                sort_text = "None"
            else:
                arrow = get_sort_indicator(self.sort_column, self.sort_column, self.sort_ascending)
                col_display = "Fill%" if self.sort_column == "fill_pct" else "Age"
                sort_text = f"{col_display}{arrow}"

            # Show both sort state and Enter hint
            indicator.update(f"[S] Sort: {sort_text}  │  [Enter] Details")
        except Exception:
            pass

    def action_cycle_sort(self) -> None:
        """Cycle through sort columns: None → Fill% ↑ → Fill% ↓ → Age ↑ → Age ↓ → None."""
        if self.sort_column is None:
            # Start sorting by fill_pct ascending
            self.sort_column = "fill_pct"
            self.sort_ascending = True
        elif self.sort_ascending:
            # Toggle to descending
            self.sort_ascending = False
        else:
            # Move to next column or back to None
            current_idx = self._sortable_columns.index(self.sort_column)
            next_idx = current_idx + 1
            if next_idx >= len(self._sortable_columns):
                # Back to no sort
                self.sort_column = None
                self.sort_ascending = True
            else:
                self.sort_column = self._sortable_columns[next_idx]
                self.sort_ascending = True

        # Notify user
        if self.sort_column is None:
            self.notify("Sort cleared", timeout=2)
        else:
            col_name = "Fill%" if self.sort_column == "fill_pct" else "Age"
            direction = "↑" if self.sort_ascending else "↓"
            self.notify(f"Sorted by {col_name} {direction}", timeout=2)

    def action_show_order_detail(self) -> None:
        """Open order detail modal for the selected order."""
        from gpt_trader.tui.widgets.portfolio.order_detail_modal import OrderDetailModal

        try:
            table = self.query_one("#orders-table", DataTable)
            if table.row_count == 0:
                self.notify("No orders to view", timeout=2)
                return

            # Get selected row key (order_id)
            cursor_row = table.cursor_row
            if cursor_row is None or cursor_row < 0:
                self.notify("Select an order first", timeout=2)
                return

            # Get the row key at cursor position
            row_key = table.get_row_at(cursor_row)
            if row_key is None:
                return

            # Find the order by ID from cached data
            order_id = str(list(table.rows.keys())[cursor_row])
            order = next(
                (row["order"] for row in self._orders_data if row["order"].order_id == order_id),
                None,
            )

            if order is None:
                self.notify("Order not found", timeout=2)
                return

            # Open the detail modal with tui_state for decision linkage
            tui_state = getattr(self.app, "tui_state", None)
            self.app.push_screen(OrderDetailModal(order, trades=self._trades, tui_state=tui_state))

        except Exception as e:
            logger.debug(f"Error opening order detail: {e}")
            self.notify("Could not open order details", timeout=2)

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
        # Store trades for detail modal access
        self._trades = trades or []

        # Build and cache fill index from trades
        self._fill_index = _build_order_fill_index(trades) if trades else {}

        # Build sortable data rows
        self._orders_data = []
        for order in orders:
            filled_qty, _ = self._get_fill_info(order, self._fill_index)
            fill_pct = self._calculate_fill_pct(order.quantity, filled_qty)
            age = get_age_seconds(order.creation_time)

            self._orders_data.append(
                {
                    "order": order,
                    "fill_pct": fill_pct,
                    "age": age if age is not None else float("inf"),  # Sort missing age to end
                }
            )

        # Refresh table with sorted data
        self._refresh_table()

    def _calculate_fill_pct(self, total_qty: Decimal, filled_qty: Decimal) -> float:
        """Calculate fill percentage as float for sorting.

        Args:
            total_qty: Total order quantity.
            filled_qty: Filled quantity.

        Returns:
            Fill percentage as float (0.0 to 100.0).
        """
        if total_qty <= 0:
            return 0.0
        return float(filled_qty) / float(total_qty) * 100.0

    def _refresh_table(self) -> None:
        """Re-sort and refresh the table with cached order data."""
        try:
            table = self.query_one("#orders-table", DataTable)
            empty_state = self.query_one("#orders-empty", TileEmptyState)
        except Exception:
            return

        # Show empty state or data
        if not self._orders_data:
            if table.row_count > 0:
                table.clear()
            table.display = False
            empty_state.display = True
            return

        table.display = True
        empty_state.display = False

        # Sort data if sort column is set
        sorted_data = self._orders_data
        if self.sort_column is not None:
            sorted_data = sort_table_data(
                self._orders_data,
                self.sort_column,
                self.sort_ascending,
                numeric_columns={"fill_pct", "age"},
            )

        # Get current row keys
        existing_keys = set(table.rows.keys())
        new_keys = {row["order"].order_id for row in sorted_data}

        # Remove orders no longer present (filled/cancelled)
        for key in existing_keys - new_keys:
            try:
                table.remove_row(key)
                logger.debug(f"Removed order row: {key}")
            except Exception:
                pass

        # Add/update orders
        for row in sorted_data:
            order = row["order"]
            row_data = self._format_order_row(order, self._fill_index)

            if order.order_id in existing_keys:
                try:
                    self._update_row_cells(table, order.order_id, row_data)
                except Exception:
                    try:
                        table.remove_row(order.order_id)
                    except Exception:
                        pass
                    table.add_row(*row_data, key=order.order_id)
            else:
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

        # Format status with color based on severity
        status_display = self._format_order_status(order.status)

        return (
            order.symbol,
            formatted_side,  # Preserves color markup
            Text(format_quantity(order.quantity), justify="right"),
            Text(format_price(order.price), justify="right"),
            Text(filled_pct, justify="right"),
            avg_px_display,
            age_display,
            status_display,
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

        Uses centralized thresholds from gpt_trader.tui.thresholds.

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

        # Color based on centralized age thresholds
        status = get_order_age_status(age)
        if status == StatusLevel.CRITICAL:
            color = THEME.colors.error
        elif status == StatusLevel.WARNING:
            color = THEME.colors.warning
        else:
            color = THEME.colors.text_muted

        return Text(age_str, style=color, justify="right")

    def _format_order_status(self, status: str) -> Text:
        """Format order status with color coding based on severity.

        Uses centralized status classification from gpt_trader.tui.thresholds.
        - OK (green): OPEN, PENDING, FILLED, CANCELLED
        - WARNING (yellow): PARTIAL, EXPIRED
        - CRITICAL (red): REJECTED, FAILED

        Args:
            status: Order status string.

        Returns:
            Rich Text with colored status display.
        """
        status_level = get_order_status_level(status)
        color = get_status_color(status_level)

        return Text(status, style=color)

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
