"""Common parsing helpers for normalising primitive configuration values."""

from __future__ import annotations

from enum import Enum
from typing import Final, TypeVar

EnumT = TypeVar("EnumT", bound=Enum)

TRUE_BOOLEAN_TOKENS: Final[frozenset[str]] = frozenset({"1", "true", "t", "yes", "y", "on"})
FALSE_BOOLEAN_TOKENS: Final[frozenset[str]] = frozenset({"0", "false", "f", "no", "n", "off"})


def interpret_tristate_bool(value: str | None) -> bool | None:
    """Return ``True``/``False`` for recognised boolean tokens otherwise ``None``.

    The helper treats empty or whitespace-only strings the same as ``None`` so callers
    can pass-through unset environment variables without additional sanitisation.
    """

    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None

    lowered = normalized.lower()
    if lowered in TRUE_BOOLEAN_TOKENS:
        return True
    if lowered in FALSE_BOOLEAN_TOKENS:
        return False
    return None


def coerce_enum(
    value: str | EnumT,
    enum_class: type[EnumT],
    *,
    case_sensitive: bool = False,
    aliases: dict[str, EnumT] | None = None,
) -> tuple[EnumT | None, str]:
    """Coerce a string or enum value to an enum, returning both enum and string.

    This utility consolidates the repetitive enum coercion pattern found throughout
    the codebase (e.g., OrderSide, OrderType, TimeInForce conversions).

    Args:
        value: String or enum value to coerce.
        enum_class: The target enum class.
        case_sensitive: If False (default), strings are uppercased before matching.
        aliases: Optional mapping of string aliases to enum values (e.g., {"GTD": TimeInForce.GTC}).

    Returns:
        A tuple of (enum_value, string_value) where:
        - enum_value is the resolved enum or None if coercion failed
        - string_value is always the normalized string representation

    Examples:
        >>> coerce_enum("buy", OrderSide)
        (OrderSide.BUY, "BUY")

        >>> coerce_enum(OrderSide.SELL, OrderSide)
        (OrderSide.SELL, "SELL")

        >>> coerce_enum("invalid", OrderSide)
        (None, "INVALID")

        >>> coerce_enum("gtd", TimeInForce, aliases={"GTD": TimeInForce.GTC})
        (TimeInForce.GTC, "GTD")
    """
    # If already an enum of the correct type, return it
    if isinstance(value, enum_class):
        return value, value.value

    # Convert string to normalized form
    string_value = value if case_sensitive else value.upper()

    # Check aliases first
    if aliases and string_value in aliases:
        enum_value = aliases[string_value]
        return enum_value, string_value

    # Try to coerce to enum
    try:
        enum_value = enum_class(string_value)
        return enum_value, enum_value.value
    except ValueError:
        return None, string_value


__all__ = [
    "FALSE_BOOLEAN_TOKENS",
    "TRUE_BOOLEAN_TOKENS",
    "interpret_tristate_bool",
    "coerce_enum",
    "EnumT",
]
