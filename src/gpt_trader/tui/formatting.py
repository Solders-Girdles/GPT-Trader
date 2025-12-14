"""
Formatting utilities for TUI display.

Handles display formatting of Decimal values for prices, quantities, P&L, etc.
Presentation layer only - does not perform calculations.
"""

from decimal import Decimal


def format_currency(value: Decimal, decimals: int = 2) -> str:
    """
    Format currency value with $ and commas.

    Args:
        value: Decimal value to format
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted string like "$1,234.56"

    Examples:
        >>> format_currency(Decimal("1234.56"))
        '$1,234.56'
        >>> format_currency(Decimal("0.5"), decimals=4)
        '$0.5000'
    """
    return f"${value:,.{decimals}f}"


def format_price(value: Decimal, decimals: int = 4) -> str:
    """
    Format price with variable decimals (no $ sign).

    Args:
        value: Decimal price to format
        decimals: Number of decimal places (default: 4 for crypto)

    Returns:
        Formatted string like "1,234.5678"

    Examples:
        >>> format_price(Decimal("1234.5678"))
        '1,234.5678'
        >>> format_price(Decimal("0.00012345"), decimals=8)
        '0.00012345'
    """
    return f"{value:,.{decimals}f}"


def format_quantity(value: Decimal, decimals: int = 8) -> str:
    """
    Format quantity (crypto can have many decimals).

    Removes trailing zeros for cleaner display.

    Args:
        value: Decimal quantity to format
        decimals: Maximum decimal places (default: 8)

    Returns:
        Formatted string with trailing zeros removed

    Examples:
        >>> format_quantity(Decimal("1.50000000"))
        '1.5'
        >>> format_quantity(Decimal("0.00012345"))
        '0.00012345'
    """
    # Format with max decimals, then strip trailing zeros
    formatted = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return formatted


def format_pnl(value: Decimal, decimals: int = 2) -> str:
    """
    Format P&L with +/- sign.

    Args:
        value: Decimal P&L value
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted string like "+123.45" or "-67.89"

    Examples:
        >>> format_pnl(Decimal("123.45"))
        '+123.45'
        >>> format_pnl(Decimal("-67.89"))
        '-67.89'
        >>> format_pnl(Decimal("0"))
        '+0.00'
    """
    return f"{value:+,.{decimals}f}"


def format_percentage(value: Decimal, decimals: int = 2) -> str:
    """
    Format percentage with % sign.

    Args:
        value: Decimal percentage value (e.g., 5.67 for 5.67%)
        decimals: Number of decimal places (default: 2)

    Returns:
        Formatted string like "+5.67%" or "-2.34%"

    Examples:
        >>> format_percentage(Decimal("5.67"))
        '+5.67%'
        >>> format_percentage(Decimal("-2.34"))
        '-2.34%'
    """
    return f"{value:+.{decimals}f}%"


def safe_decimal(value: str | int | float | Decimal, default: Decimal = Decimal("0")) -> Decimal:
    """
    Safely convert any numeric type to Decimal.

    Args:
        value: Value to convert (str, int, float, or Decimal)
        default: Default value if conversion fails

    Returns:
        Decimal value or default on error

    Examples:
        >>> safe_decimal("123.45")
        Decimal('123.45')
        >>> safe_decimal("invalid", Decimal("0"))
        Decimal('0')
        >>> safe_decimal(123)
        Decimal('123')
    """
    if isinstance(value, Decimal):
        return value

    try:
        if isinstance(value, str):
            # Clean common formatting
            cleaned = value.replace("$", "").replace(",", "").replace("%", "").strip()
            if not cleaned or cleaned in ("N/A", "n/a", "-"):
                return default
            return Decimal(cleaned)
        return Decimal(str(value))
    except (ValueError, TypeError):
        return default
