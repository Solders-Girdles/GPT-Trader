"""
Simplified adapters for normalizing inputs to core types.

This module only provides helpers to convert strings and basic types
to core enums and Decimal values. No conversions back to local types.
"""

from decimal import Decimal

from ..brokerages.core.interfaces import OrderSide as CoreOrderSide

# Import core interfaces
from ..brokerages.core.interfaces import OrderType as CoreOrderType
from ..brokerages.core.interfaces import TimeInForce as CoreTimeInForce


def to_core_tif(tif_str: str | CoreTimeInForce) -> CoreTimeInForce:
    """
    Convert string or enum time-in-force to core enum.

    Args:
        tif_str: String like 'day', 'gtc', 'ioc', 'fok' or CoreTimeInForce enum

    Returns:
        CoreTimeInForce enum value
    """
    # If already a CoreTimeInForce enum, return it
    if isinstance(tif_str, CoreTimeInForce):
        return tif_str

    # Handle string values
    tif_map = {
        "day": CoreTimeInForce.GTC,  # Map 'day' to GTC
        "gtc": CoreTimeInForce.GTC,
        "ioc": CoreTimeInForce.IOC,
        "fok": CoreTimeInForce.FOK,
    }

    # Convert to string if it's another type of enum
    tif_value = tif_str.value.lower() if hasattr(tif_str, "value") else str(tif_str).lower()
    return tif_map.get(tif_value, CoreTimeInForce.GTC)


def to_core_side(side: str | CoreOrderSide) -> CoreOrderSide:
    """
    Convert string or enum to core OrderSide.

    Args:
        side: String 'buy'/'sell' or CoreOrderSide enum

    Returns:
        CoreOrderSide enum value
    """
    if isinstance(side, CoreOrderSide):
        return side

    side_map = {"buy": CoreOrderSide.BUY, "sell": CoreOrderSide.SELL}
    return side_map.get(side.lower(), CoreOrderSide.BUY)


def to_core_type(order_type: str | CoreOrderType) -> CoreOrderType:
    """
    Convert string or enum to core OrderType.

    Args:
        order_type: String like 'market', 'limit', etc. or CoreOrderType enum

    Returns:
        CoreOrderType enum value
    """
    if isinstance(order_type, CoreOrderType):
        return order_type

    type_map = {
        "market": CoreOrderType.MARKET,
        "limit": CoreOrderType.LIMIT,
        "stop": CoreOrderType.STOP,
        "stop_limit": CoreOrderType.STOP_LIMIT,
    }
    return type_map.get(order_type.lower(), CoreOrderType.MARKET)


def to_decimal(value: str | int | float | Decimal | None) -> Decimal | None:
    """
    Convert various numeric types to Decimal for financial precision.

    Args:
        value: Numeric value or None

    Returns:
        Decimal value or None
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))
