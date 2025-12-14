"""
TUI Utilities Package.

Provides shared formatting utilities for consistent display across widgets.

Modules:
    pnl_formatting: P&L, leverage, and directional color formatting
    table_formatting: DataTable cell formatting, sorting, and clipboard helpers

Usage:
    from gpt_trader.tui.utilities import (
        # P&L and color formatting
        format_pnl_colored,
        format_side_colored,
        format_leverage_colored,
        format_direction_colored,
        # Table formatting
        format_timestamp,
        truncate_id,
        format_table_cell,
        get_sort_indicator,
        # Sorting
        sort_table_data,
        # Clipboard
        copy_to_clipboard,
        format_row_for_copy,
    )
"""

from gpt_trader.tui.utilities.pnl_formatting import (
    format_direction_colored,
    format_leverage_colored,
    format_pnl_colored,
    format_side_colored,
    get_sparkline_color,
)
from gpt_trader.tui.utilities.table_formatting import (
    copy_to_clipboard,
    create_numeric_cell,
    format_row_for_copy,
    format_table_cell,
    format_timestamp,
    get_sort_indicator,
    sort_table_data,
    truncate_id,
)

__all__ = [
    # Clipboard utilities
    "copy_to_clipboard",
    "format_row_for_copy",
    # Cell formatting
    "create_numeric_cell",
    "format_table_cell",
    "format_timestamp",
    "get_sort_indicator",
    "truncate_id",
    # P&L and color formatting
    "format_direction_colored",
    "format_leverage_colored",
    "format_pnl_colored",
    "format_side_colored",
    "get_sparkline_color",
    # Sorting
    "sort_table_data",
]
