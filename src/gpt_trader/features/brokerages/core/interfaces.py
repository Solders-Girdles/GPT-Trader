"""Broker interface types - re-exported from core for backward compatibility.

.. deprecated:: 2.0
    All types are now defined in ``gpt_trader.core``. This module re-exports them
    only for backward compatibility.

    Migration::

        # Old (deprecated)
        from gpt_trader.features.brokerages.core.interfaces import Order, Balance

        # New (preferred)
        from gpt_trader.core import Order, Balance

    Removal planned for v3.0.
"""

import warnings

# Emit deprecation warning on module import
warnings.warn(
    "Importing from gpt_trader.features.brokerages.core.interfaces is deprecated. "
    "Import directly from gpt_trader.core instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all core types for backward compatibility
from gpt_trader.core import (  # noqa: E402
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
