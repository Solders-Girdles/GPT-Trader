"""Core trading enums used across all slices."""

from enum import Enum


class OrderSide(str, Enum):
    """Order direction - buy or sell."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order execution type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    """Order time-in-force policy."""

    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill


class MarketType(str, Enum):
    """Type of market/instrument."""

    SPOT = "SPOT"
    FUTURE = "FUTURE"
    PERPETUAL = "PERPETUAL"
    FUTURES = "FUTURES"  # Alias for compatibility


class OrderStatus(str, Enum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    FAILED = "FAILED"
