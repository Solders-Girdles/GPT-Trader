"""
TUI Utilities Package.

Provides shared formatting utilities for consistent display across widgets.

Modules:
    pnl_formatting: P&L, leverage, and directional color formatting
    table_formatting: DataTable cell formatting helpers

Usage:
    from gpt_trader.tui.utilities import (
        format_pnl_colored,
        format_side_colored,
        format_leverage_colored,
        format_direction_colored,
        format_timestamp,
        truncate_id,
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
    create_numeric_cell,
    format_timestamp,
    truncate_id,
)

__all__ = [
    "create_numeric_cell",
    "format_direction_colored",
    "format_leverage_colored",
    "format_pnl_colored",
    "format_side_colored",
    "format_timestamp",
    "get_sparkline_color",
    "truncate_id",
]
