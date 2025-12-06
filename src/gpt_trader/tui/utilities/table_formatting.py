"""
DataTable formatting utilities.

Provides helpers for consistent table cell formatting across widgets.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from rich.text import Text


def format_timestamp(timestamp: str, format_str: str = "%H:%M:%S") -> str:
    """
    Format ISO timestamp to display format.

    Handles common ISO 8601 formats and extracts time portion.

    Args:
        timestamp: ISO timestamp string (e.g., "2024-01-15T10:30:45.123Z")
        format_str: Output format string (default: HH:MM:SS)

    Returns:
        Formatted time string or original if parsing fails

    Examples:
        >>> format_timestamp("2024-01-15T10:30:45.123Z")
        '10:30:45'
        >>> format_timestamp("2024-01-15T10:30:45")
        '10:30:45'
    """
    if not timestamp:
        return ""

    # Quick extraction for common ISO format
    if "T" in timestamp:
        try:
            time_part = timestamp.split("T")[1].split(".")[0].split("Z")[0]
            return time_part
        except IndexError:
            pass

    # Full datetime parsing fallback
    try:
        # Try common formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(timestamp, fmt)
                return dt.strftime(format_str)
            except ValueError:
                continue
    except Exception:
        pass

    return timestamp


def truncate_id(order_id: str, length: int = 8) -> str:
    """
    Truncate order/trade ID for compact display.

    Takes the last N characters of the ID for display.

    Args:
        order_id: Full order or trade ID string
        length: Number of characters to keep (default: 8)

    Returns:
        Truncated ID or empty string if None

    Examples:
        >>> truncate_id("abc123def456ghi789")
        'ghi789'
        >>> truncate_id("short", 8)
        'short'
    """
    if not order_id:
        return ""
    return order_id[-length:] if len(order_id) > length else order_id


def create_numeric_cell(
    value: Any,
    format_fn: callable | None = None,
    justify: str = "right",
) -> Text:
    """
    Create a right-aligned numeric cell for DataTable.

    Args:
        value: Value to format
        format_fn: Optional formatting function to apply
        justify: Text justification (default: "right")

    Returns:
        Rich Text object for table cell

    Examples:
        >>> create_numeric_cell(1234.56, format_fn=lambda x: f"${x:,.2f}")
        Text('$1,234.56', justify='right')
    """
    if format_fn:
        display_str = format_fn(value)
    else:
        display_str = str(value)

    return Text(display_str, justify=justify)
