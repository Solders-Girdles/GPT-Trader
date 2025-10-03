"""
Execution engine models and configuration.

Defines order configurations, sizing modes, and stop trigger models for the
advanced execution engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    Product,
    Quote,
    TimeInForce,
)

__all__ = [
    "SizingMode",
    "OrderConfig",
    "StopTrigger",
    "NormalizedOrderRequest",
]


class SizingMode(Enum):
    """Position sizing strategy."""

    CONSERVATIVE = "conservative"  # Downsize to fit impact limit
    STRICT = "strict"  # Reject if can't fit
    AGGRESSIVE = "aggressive"  # Allow higher impact


@dataclass
class OrderConfig:
    """Configuration for advanced order types."""

    # Limit orders
    enable_limit_orders: bool = True
    limit_price_offset_bps: Decimal = Decimal("5")  # Offset from mid

    # Stop orders
    enable_stop_orders: bool = True
    stop_trigger_offset_pct: Decimal = Decimal("0.02")  # 2% from entry

    # Stop-limit orders
    enable_stop_limit: bool = True
    stop_limit_spread_bps: Decimal = Decimal("10")

    # Post-only protection
    enable_post_only: bool = True
    reject_on_cross: bool = True  # Reject if post-only would cross

    # TIF support
    enable_ioc: bool = True
    enable_fok: bool = False  # Gate until confirmed

    # Sizing
    sizing_mode: SizingMode = SizingMode.CONSERVATIVE
    max_impact_bps: Decimal = Decimal("15")


@dataclass
class StopTrigger:
    """Stop order trigger tracking."""

    order_id: str
    symbol: str
    trigger_price: Decimal
    side: OrderSide
    quantity: Decimal
    limit_price: Decimal | None = None
    created_at: datetime = field(default_factory=datetime.now)
    triggered: bool = False
    triggered_at: datetime | None = None


@dataclass
class NormalizedOrderRequest:
    """Normalized order request ready for validation.

    This dataclass represents an order request that has been normalized with:
    - Generated/validated client_id
    - Quantity converted to Decimal
    - Market data fetched (product, quote)
    - All parameters validated for basic correctness

    Attributes:
        client_id: Unique client identifier for this order
        symbol: Trading symbol
        side: Order side (BUY/SELL)
        quantity: Order quantity (always Decimal)
        order_type: Order type (MARKET, LIMIT, STOP, etc.)
        limit_price: Limit price for LIMIT/STOP_LIMIT orders
        stop_price: Stop price for STOP/STOP_LIMIT orders
        time_in_force: Time-in-force policy
        reduce_only: Whether order can only reduce position
        post_only: Whether order should be post-only
        leverage: Leverage multiplier (for derivatives)
        product: Product information from broker (if available)
        quote: Current market quote (if post_only=True)
    """

    client_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    limit_price: Decimal | None
    stop_price: Decimal | None
    time_in_force: TimeInForce
    reduce_only: bool
    post_only: bool
    leverage: int | None

    # Market data (fetched during normalization)
    product: Product | None = None
    quote: Quote | None = None
