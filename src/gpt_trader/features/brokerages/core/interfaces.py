"""Broker interface types - re-exported from core for backward compatibility.

All types are now defined in gpt_trader.core. This module re-exports them
to maintain backward compatibility for existing imports from brokerages.core.interfaces.

Migration: Import directly from gpt_trader.core instead.
"""

# Re-export all core types for backward compatibility
from gpt_trader.core import (
    # Exceptions
    AuthError,
    # Account types
    Balance,
    BrokerageError,
    # Market data types
    Candle,
    InsufficientFunds,
    InvalidRequestError,
    # Trading enums
    MarketType,
    NotFoundError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    PermissionDeniedError,
    Position,
    Product,
    Quote,
    RateLimitError,
    TimeInForce,
)

__all__ = [
    # Trading enums
    "MarketType",
    "OrderSide",
    "OrderStatus",
    "OrderType",
    "TimeInForce",
    # Market data types
    "Candle",
    "Product",
    "Quote",
    # Account types
    "Balance",
    "Order",
    "Position",
    # Exceptions
    "AuthError",
    "BrokerageError",
    "InsufficientFunds",
    "InvalidRequestError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
]
