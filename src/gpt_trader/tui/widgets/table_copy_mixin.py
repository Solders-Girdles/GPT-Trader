"""
DataTable copy functionality mixin.

Provides keyboard shortcuts for copying table rows to clipboard.
Mix this into widgets that contain DataTable elements.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.binding import Binding
from textual.widgets import DataTable

from gpt_trader.tui.notification_helpers import notify_success, notify_warning
from gpt_trader.tui.utilities.table_formatting import copy_to_clipboard
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__, component="tui")


class TableCopyMixin:
    """Mixin providing copy-to-clipboard functionality for DataTable widgets.

    Add this mixin to any widget containing a DataTable to enable:
    - Press 'c' to copy the currently selected row
    - Press 'C' (shift+c) to copy all visible rows

    Usage:
        class MyTableWidget(TableCopyMixin, Static):
            BINDINGS = [
                *TableCopyMixin.COPY_BINDINGS,
                # ... other bindings
            ]

            def get_table(self) -> DataTable:
                return self.query_one(DataTable)
    """

    COPY_BINDINGS = [
        Binding("c", "copy_row", "Copy", show=True),
        Binding("C", "copy_all", "Copy All", show=True),
    ]

    def get_table(self) -> DataTable | None:
        """Override to return the DataTable widget.

        Returns:
            The DataTable widget to copy from, or None if not available.
        """
        try:
            return self.query_one(DataTable)
        except Exception:
            return None

    def get_copy_columns(self) -> list[str] | None:
        """Override to specify which columns to copy.

        Returns:
            List of column keys to include in copy, or None for all columns.
        """
        return None

    def action_copy_row(self) -> None:
        """Copy the currently selected row to clipboard."""
        table = self.get_table()
        if not table:
            return

        # Get current cursor row
        if table.cursor_row is None or table.row_count == 0:
            notify_warning(self, "No row selected")
            return

        try:
            row_key = list(table.rows.keys())[table.cursor_row]
            row_data = self._extract_row_data(table, row_key)

            if row_data:
                text = "\t".join(str(v) for v in row_data)
                if copy_to_clipboard(text):
                    notify_success(self, "Row copied to clipboard")
                else:
                    notify_warning(self, "Could not access clipboard")
            else:
                notify_warning(self, "No data to copy")

        except Exception as e:
            logger.debug(f"Copy row failed: {e}")
            notify_warning(self, "Copy failed")

    def action_copy_all(self) -> None:
        """Copy all visible rows to clipboard."""
        table = self.get_table()
        if not table or table.row_count == 0:
            notify_warning(self, "No data to copy")
            return

        try:
            # Get column headers
            columns = list(table.columns.keys())
            header_labels = [str(table.columns[col].label) for col in columns]
            lines = ["\t".join(header_labels)]

            # Get all rows
            for row_key in table.rows.keys():
                row_data = self._extract_row_data(table, row_key)
                if row_data:
                    lines.append("\t".join(str(v) for v in row_data))

            text = "\n".join(lines)
            if copy_to_clipboard(text):
                notify_success(self, f"Copied {table.row_count} rows to clipboard")
            else:
                notify_warning(self, "Could not access clipboard")

        except Exception as e:
            logger.debug(f"Copy all failed: {e}")
            notify_warning(self, "Copy failed")

    def _extract_row_data(self, table: DataTable, row_key: Any) -> list[str]:
        """Extract data from a table row.

        Args:
            table: The DataTable to extract from.
            row_key: The row key to extract.

        Returns:
            List of string values from the row.
        """
        values = []
        columns = list(table.columns.keys())

        for col_key in columns:
            try:
                cell = table.get_cell(row_key, col_key)
                # Handle Rich Text objects
                if hasattr(cell, "plain"):
                    values.append(cell.plain)
                else:
                    values.append(str(cell) if cell is not None else "")
            except Exception:
                values.append("")

        return values


# Column header standards for consistency
STANDARD_HEADERS = {
    # Positions table
    "symbol": "Symbol",
    "type": "Type",
    "side": "Side",
    "quantity": "Qty",
    "entry_price": "Entry",
    "mark_price": "Mark",
    "pnl": "P&L",
    "pnl_pct": "%",
    "leverage": "Lev",
    "liquidation": "Liq%",
    # Trades table
    "time": "Time",
    "order_id": "Order",
    "price": "Price",
    # Account table
    "asset": "Asset",
    "total": "Total",
    "available": "Avail",
    # Common
    "status": "Status",
    "action": "Action",
    "category": "Cat",
    "severity": "Sev",
    "message": "Message",
    "title": "Title",
}


def get_standard_header(column_key: str) -> str:
    """Get standardized header label for a column.

    Args:
        column_key: The column key or name.

    Returns:
        Standardized header label.
    """
    key = column_key.lower().replace(" ", "_").replace("-", "_")
    return STANDARD_HEADERS.get(key, column_key)
