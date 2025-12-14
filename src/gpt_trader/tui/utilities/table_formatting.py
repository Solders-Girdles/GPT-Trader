"""
DataTable formatting utilities.

Provides helpers for consistent table cell formatting across widgets:
- Cell formatting and alignment
- Data sorting with type awareness
- Clipboard operations (cross-platform)
- Timestamp and ID truncation
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from decimal import Decimal
from typing import Any

from rich.text import Text

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


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


# =============================================================================
# Sorting Utilities
# =============================================================================


def sort_table_data(
    data: list[dict[str, Any]],
    column: str,
    ascending: bool = True,
    numeric_columns: set[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Sort table data by column with type awareness.

    Handles numeric sorting for columns containing:
    - Numbers (int, float, Decimal)
    - Strings with currency/percent symbols ($1,234.56, 5.2%)
    - None/empty values (sorted to end)

    Args:
        data: List of row dictionaries to sort.
        column: Column key to sort by.
        ascending: Sort direction (True = A-Z/low-high, False = Z-A/high-low).
        numeric_columns: Set of column keys that should sort numerically.
                        If None, attempts auto-detection.

    Returns:
        Sorted copy of the data list.

    Examples:
        >>> data = [{"name": "BTC", "price": 50000}, {"name": "ETH", "price": 3000}]
        >>> sort_table_data(data, "price", ascending=False, numeric_columns={"price"})
        [{"name": "BTC", "price": 50000}, {"name": "ETH", "price": 3000}]
    """
    if not data:
        return data

    numeric_cols = numeric_columns or set()

    # Auto-detect numeric columns if not specified
    if not numeric_cols:
        numeric_cols = _detect_numeric_columns(data, column)

    is_numeric = column in numeric_cols

    def sort_key(row: dict[str, Any]) -> tuple[int, Any]:
        """
        Generate sort key with None values sorted to end.

        Returns tuple: (priority, value) where priority ensures
        None/empty values appear last regardless of sort direction.
        """
        value = row.get(column)

        # Handle None/empty - always sort to end
        if value is None or value == "":
            return (1, 0 if is_numeric else "")

        if is_numeric:
            try:
                return (0, _parse_numeric(value))
            except (ValueError, TypeError):
                return (1, 0)

        # String comparison (case-insensitive)
        return (0, str(value).lower())

    return sorted(data, key=sort_key, reverse=not ascending)


def _parse_numeric(value: Any) -> float:
    """
    Parse various numeric formats to float.

    Handles:
    - int, float, Decimal
    - Strings with $, %, commas
    - Rich Text objects

    Args:
        value: Value to parse.

    Returns:
        Parsed float value.

    Raises:
        ValueError: If value cannot be parsed as numeric.
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, Text):
        value = value.plain

    if isinstance(value, str):
        # Strip currency symbols, commas, percent signs
        cleaned = value.replace("$", "").replace(",", "").replace("%", "").strip()

        # Handle parentheses for negative (accounting format)
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]

        return float(cleaned)

    raise ValueError(f"Cannot parse {type(value)} as numeric")


def _detect_numeric_columns(data: list[dict[str, Any]], column: str) -> set[str]:
    """
    Auto-detect if a column contains numeric data.

    Samples first few rows to determine column type.

    Args:
        data: Data rows to sample.
        column: Column key to check.

    Returns:
        Set containing column key if numeric, empty set otherwise.
    """
    sample_size = min(5, len(data))
    numeric_count = 0

    for row in data[:sample_size]:
        value = row.get(column)
        if value is None or value == "":
            continue

        try:
            _parse_numeric(value)
            numeric_count += 1
        except (ValueError, TypeError):
            pass

    # Consider numeric if most samples parse successfully
    if numeric_count > sample_size / 2:
        return {column}

    return set()


# =============================================================================
# Clipboard Utilities
# =============================================================================


def copy_to_clipboard(text: str) -> bool:
    """
    Copy text to system clipboard (cross-platform).

    Supports:
    - macOS: pbcopy
    - Linux: xclip (with clipboard selection)
    - Windows: clip
    - Fallback: pyperclip library if available

    Args:
        text: Text to copy to clipboard.

    Returns:
        True if copy succeeded, False otherwise.

    Examples:
        >>> copy_to_clipboard("Hello, World!")
        True
    """
    try:
        if sys.platform == "darwin":
            # macOS
            subprocess.run(
                ["pbcopy"],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
            )
            return True

        elif sys.platform.startswith("linux"):
            # Linux with xclip
            subprocess.run(
                ["xclip", "-selection", "clipboard"],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
            )
            return True

        elif sys.platform == "win32":
            # Windows
            subprocess.run(
                ["clip"],
                input=text.encode("utf-8"),
                check=True,
                capture_output=True,
                shell=True,
            )
            return True

    except FileNotFoundError:
        logger.debug("Clipboard tool not found, trying pyperclip")
    except subprocess.CalledProcessError as e:
        logger.debug(f"Clipboard subprocess failed: {e}")
    except Exception as e:
        logger.debug(f"Clipboard copy failed: {e}")

    # Fallback: try pyperclip if available
    try:
        import pyperclip

        pyperclip.copy(text)
        return True
    except ImportError:
        logger.debug("pyperclip not installed")
    except Exception as e:
        logger.debug(f"pyperclip copy failed: {e}")

    return False


def format_row_for_copy(
    row: dict[str, Any],
    columns: list[str],
    separator: str = "\t",
) -> str:
    """
    Format a row dictionary as a copyable string.

    Args:
        row: Row data dictionary.
        columns: Column keys in display order.
        separator: Field separator (default: tab for spreadsheet paste).

    Returns:
        Formatted string suitable for clipboard.

    Examples:
        >>> format_row_for_copy({"a": 1, "b": 2}, ["a", "b"])
        "1\t2"
    """
    values = []
    for col in columns:
        value = row.get(col, "")
        # Convert Rich Text to plain string
        if isinstance(value, Text):
            value = value.plain
        values.append(str(value) if value is not None else "")

    return separator.join(values)


# =============================================================================
# Cell Formatting Utilities
# =============================================================================


def format_table_cell(
    value: Any,
    column_type: str = "text",
    decimal_places: int = 2,
    max_length: int | None = None,
) -> str:
    """
    Format cell value based on column type.

    Provides consistent formatting across tables for common data types.

    Args:
        value: Raw cell value.
        column_type: Type hint for formatting:
            - "text": Default string conversion
            - "number": Numeric with thousand separators
            - "currency": Currency format ($X,XXX.XX)
            - "percent": Percentage format (X.XX%)
            - "timestamp": Time portion only (HH:MM:SS)
            - "id": Truncated ID (last 8 chars)
        decimal_places: Decimal precision for numeric types (default: 2).
        max_length: Maximum string length (truncate with ellipsis if exceeded).

    Returns:
        Formatted string for display.

    Examples:
        >>> format_table_cell(1234567.89, "currency")
        "$1,234,567.89"
        >>> format_table_cell(0.1523, "percent")
        "15.23%"
    """
    if value is None:
        return "--"

    if isinstance(value, Text):
        value = value.plain

    result: str

    if column_type == "number":
        try:
            num = float(value) if not isinstance(value, (int, float, Decimal)) else value
            result = f"{num:,.{decimal_places}f}"
        except (ValueError, TypeError):
            result = str(value)

    elif column_type == "currency":
        try:
            num = float(value) if not isinstance(value, (int, float, Decimal)) else value
            result = f"${num:,.{decimal_places}f}"
        except (ValueError, TypeError):
            result = str(value)

    elif column_type == "percent":
        try:
            num = float(value) if not isinstance(value, (int, float, Decimal)) else value
            # Assume value is already in decimal form (0.15 = 15%)
            result = f"{num * 100:.{decimal_places}f}%"
        except (ValueError, TypeError):
            result = str(value)

    elif column_type == "timestamp":
        result = format_timestamp(str(value))

    elif column_type == "id":
        result = truncate_id(str(value))

    else:  # "text" or default
        result = str(value)

    # Apply max length truncation
    if max_length and len(result) > max_length:
        result = result[: max_length - 1] + "…"

    return result


def get_sort_indicator(column_key: str, sort_column: str | None, ascending: bool) -> str:
    """
    Get sort direction indicator for column header.

    Args:
        column_key: Column to check.
        sort_column: Currently sorted column (or None).
        ascending: Sort direction.

    Returns:
        Sort indicator string (" ▲", " ▼", or "").

    Examples:
        >>> get_sort_indicator("price", "price", True)
        " ▲"
        >>> get_sort_indicator("price", "price", False)
        " ▼"
        >>> get_sort_indicator("price", "name", True)
        ""
    """
    if column_key != sort_column:
        return ""

    return " ▲" if ascending else " ▼"
