"""
P&L and directional color formatting utilities.

Provides consistent color coding for:
- P&L values (positive=green, negative=red, neutral=dim)
- Trade sides (BUY=green, SELL=red)
- Leverage levels (low=green, medium=yellow, high=red)
- Direction indicators (up=green, down=red)

These utilities use the TUI theme for consistent color application.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from rich.text import Text

from gpt_trader.tui.theme import THEME

if TYPE_CHECKING:
    pass


def format_pnl_colored(
    value: float | Decimal | None,
    display_str: str | None = None,
    justify: str = "right",
) -> Text:
    """
    Format P&L value with appropriate color coding.

    Colors:
        - Positive: success/green
        - Negative: error/red
        - Zero/None: neutral/dim

    Args:
        value: Numeric P&L value (can be None for N/A)
        display_str: Optional display string (if None, uses str(value))
        justify: Text justification ("left", "center", "right")

    Returns:
        Rich Text object with color styling

    Examples:
        >>> format_pnl_colored(100.50, "$100.50")  # Green
        >>> format_pnl_colored(-50.25, "-$50.25")  # Red
        >>> format_pnl_colored(None, "N/A")        # Dim
    """
    if display_str is None:
        display_str = "N/A" if value is None else str(value)

    if value is None:
        return Text(display_str, style="dim", justify=justify)
    elif value > 0:
        return Text.from_markup(
            f"[{THEME.colors.success}]{display_str}[/{THEME.colors.success}]",
            justify=justify,
        )
    elif value < 0:
        return Text.from_markup(
            f"[{THEME.colors.error}]{display_str}[/{THEME.colors.error}]",
            justify=justify,
        )
    else:
        # Zero - neutral
        return Text(display_str, justify=justify)


def format_side_colored(side: str) -> str:
    """
    Format trade side (BUY/SELL) with color markup.

    Colors:
        - BUY: success/green
        - SELL: error/red

    Args:
        side: Trade side string ("BUY" or "SELL")

    Returns:
        Rich markup string with color

    Examples:
        >>> format_side_colored("BUY")   # "[green]BUY[/green]"
        >>> format_side_colored("SELL")  # "[red]SELL[/red]"
    """
    color = THEME.colors.success if side == "BUY" else THEME.colors.error
    return f"[{color}]{side}[/{color}]"


def format_leverage_colored(
    leverage: float,
    justify: str = "right",
    low_threshold: float = 2.0,
    medium_threshold: float = 5.0,
) -> Text:
    """
    Format leverage value with risk-based color coding.

    Colors:
        - Low (<2x): green (safe)
        - Medium (2-5x): yellow (caution)
        - High (>5x): red (risky)

    Args:
        leverage: Leverage multiplier value
        justify: Text justification ("left", "center", "right")
        low_threshold: Upper bound for low risk (default: 2.0)
        medium_threshold: Upper bound for medium risk (default: 5.0)

    Returns:
        Rich Text object with color styling

    Examples:
        >>> format_leverage_colored(1.5)   # Green "1.5x"
        >>> format_leverage_colored(3.0)   # Yellow "3.0x"
        >>> format_leverage_colored(10.0)  # Red "10.0x"
    """
    leverage_str = f"{leverage:.1f}x"

    if leverage < low_threshold:
        markup = f"[green]{leverage_str}[/green]"
    elif leverage < medium_threshold:
        markup = f"[yellow]{leverage_str}[/yellow]"
    else:
        markup = f"[red]{leverage_str}[/red]"

    return Text.from_markup(markup, justify=justify)


def format_direction_colored(
    value: float | Decimal,
    display_str: str,
    justify: str = "right",
) -> Text:
    """
    Format directional value (up/down) with color coding.

    Colors:
        - Positive/Up: success/green
        - Negative/Down: error/red
        - Zero: neutral (no color)

    Args:
        value: Numeric value indicating direction
        display_str: String to display (e.g., "+5.2%", "-3.1%")
        justify: Text justification ("left", "center", "right")

    Returns:
        Rich Text object with color styling

    Examples:
        >>> format_direction_colored(5.2, "+5.2%")   # Green
        >>> format_direction_colored(-3.1, "-3.1%")  # Red
    """
    if value > 0:
        return Text.from_markup(
            f"[{THEME.colors.success}]{display_str}[/{THEME.colors.success}]",
            justify=justify,
        )
    elif value < 0:
        return Text.from_markup(
            f"[{THEME.colors.error}]{display_str}[/{THEME.colors.error}]",
            justify=justify,
        )
    else:
        return Text(display_str, justify=justify)


def get_sparkline_color(values: list[float]) -> str:
    """
    Determine sparkline color based on trend direction.

    Args:
        values: List of numeric values (first = oldest, last = newest)

    Returns:
        Color name string ("green", "red", or theme success/error)

    Examples:
        >>> get_sparkline_color([1, 2, 3])  # Returns success color (uptrend)
        >>> get_sparkline_color([3, 2, 1])  # Returns error color (downtrend)
    """
    if not values or len(values) < 2:
        return THEME.colors.success

    return THEME.colors.success if values[-1] >= values[0] else THEME.colors.error
