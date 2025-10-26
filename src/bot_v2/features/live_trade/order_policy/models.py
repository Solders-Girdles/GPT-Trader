"""Order policy data structures and enums."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, TypedDict, cast


class OrderTypeSupport(Enum):
    """Order type support levels."""

    SUPPORTED = "supported"
    GATED = "gated"
    UNSUPPORTED = "unsupported"


class TIFSupport(Enum):
    """Time-in-force support levels."""

    SUPPORTED = "supported"
    GATED = "gated"
    UNSUPPORTED = "unsupported"


@dataclass
class OrderCapability:
    """Order type capability definition."""

    order_type: str  # "MARKET", "LIMIT", "STOP", "STOP_LIMIT", "BRACKET"
    tif: str  # "GTC", "IOC", "FOK", "GTD"
    support_level: OrderTypeSupport
    min_quantity: Decimal | None = None
    max_quantity: Decimal | None = None
    quantity_increment: Decimal | None = None
    price_increment: Decimal | None = None

    # Special flags
    post_only_supported: bool = True
    reduce_only_supported: bool = True
    bracket_supported: bool = False

    # Risk limits
    max_notional: Decimal | None = None
    rate_limit_per_minute: int = 60

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "order_type": self.order_type,
            "tif": self.tif,
            "support_level": self.support_level.value,
            "min_quantity": float(self.min_quantity) if self.min_quantity else None,
            "max_quantity": float(self.max_quantity) if self.max_quantity else None,
            "quantity_increment": (
                float(self.quantity_increment) if self.quantity_increment else None
            ),
            "price_increment": float(self.price_increment) if self.price_increment else None,
            "post_only_supported": self.post_only_supported,
            "reduce_only_supported": self.reduce_only_supported,
            "bracket_supported": self.bracket_supported,
            "max_notional": float(self.max_notional) if self.max_notional else None,
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }


@dataclass
class SymbolPolicy:
    """Trading policy for a specific symbol."""

    symbol: str
    environment: str  # "paper", "sandbox", "live"

    # Supported capabilities
    capabilities: list[OrderCapability] = field(default_factory=list)

    # Symbol-specific limits
    min_order_size: Decimal = Decimal("0.001")
    max_order_size: Decimal | None = None
    size_increment: Decimal = Decimal("0.001")
    price_increment: Decimal = Decimal("0.01")

    # Risk limits
    max_position_size: Decimal | None = None
    max_daily_volume: Decimal | None = None

    # Operational settings
    trading_enabled: bool = True
    reduce_only_mode: bool = False

    # Market conditions
    requires_post_only: bool = False
    spread_threshold_bps: Decimal | None = None  # Require post-only if spread > threshold

    def get_capability(self, order_type: str, tif: str) -> OrderCapability | None:
        """Get capability for specific order type and TIF."""
        for cap in self.capabilities:
            if cap.order_type == order_type and cap.tif == tif:
                return cap
        return None

    def is_order_allowed(
        self, order_type: str, tif: str, quantity: Decimal, price: Decimal | None = None
    ) -> tuple[bool, str]:
        """
        Check if order is allowed under current policy.

        Returns:
            Tuple of (allowed, reason)
        """
        if not self.trading_enabled:
            return False, "Trading disabled for symbol"

        capability = self.get_capability(order_type, tif)
        if not capability:
            return False, f"Order type {order_type} with TIF {tif} not supported"

        if capability.support_level == OrderTypeSupport.UNSUPPORTED:
            return False, f"Order type {order_type} unsupported"

        if capability.support_level == OrderTypeSupport.GATED:
            return False, f"Order type {order_type} currently gated"

        if quantity < self.min_order_size:
            return False, f"Quantity {quantity} below minimum {self.min_order_size}"

        if self.max_order_size and quantity > self.max_order_size:
            return False, f"Quantity {quantity} exceeds maximum {self.max_order_size}"

        if capability.min_quantity and quantity < capability.min_quantity:
            return False, f"Quantity {quantity} below capability minimum {capability.min_quantity}"

        if capability.max_quantity and quantity > capability.max_quantity:
            return (
                False,
                f"Quantity {quantity} exceeds capability maximum {capability.max_quantity}",
            )

        remainder = quantity % self.size_increment
        if remainder != 0:
            return False, f"Quantity {quantity} not aligned to increment {self.size_increment}"

        if price and self.price_increment:
            price_remainder = price % self.price_increment
            if price_remainder != 0:
                return False, f"Price {price} not aligned to increment {self.price_increment}"

        if price and capability.max_notional:
            notional = quantity * price
            if notional > capability.max_notional:
                return False, f"Notional {notional} exceeds maximum {capability.max_notional}"

        return True, "Order allowed"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "symbol": self.symbol,
            "environment": self.environment,
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "min_order_size": float(self.min_order_size),
            "max_order_size": float(self.max_order_size) if self.max_order_size else None,
            "size_increment": float(self.size_increment),
            "price_increment": float(self.price_increment),
            "max_position_size": float(self.max_position_size) if self.max_position_size else None,
            "max_daily_volume": float(self.max_daily_volume) if self.max_daily_volume else None,
            "trading_enabled": self.trading_enabled,
            "reduce_only_mode": self.reduce_only_mode,
            "requires_post_only": self.requires_post_only,
            "spread_threshold_bps": (
                float(self.spread_threshold_bps) if self.spread_threshold_bps else None
            ),
        }


class OrderConfig(TypedDict, total=False):
    order_type: str
    tif: str
    post_only: bool
    reduce_only: bool
    use_market: bool
    fallback_reason: str
    error: str


class SupportedOrderConfig(TypedDict):
    order_type: str
    tif: str
    post_only: bool
    reduce_only: bool


def cast_supported_capabilities(
    capabilities: list[OrderCapability],
) -> list[SupportedOrderConfig]:
    """Helper to convert supported capabilities into dict structures."""
    supported: list[SupportedOrderConfig] = []
    for capability in capabilities:
        if capability.support_level == OrderTypeSupport.SUPPORTED:
            supported.append(
                cast(
                    SupportedOrderConfig,
                    {
                        "order_type": capability.order_type,
                        "tif": capability.tif,
                        "post_only": capability.post_only_supported,
                        "reduce_only": capability.reduce_only_supported,
                    },
                )
            )
    return supported


__all__ = [
    "OrderCapability",
    "OrderConfig",
    "OrderTypeSupport",
    "SupportedOrderConfig",
    "SymbolPolicy",
    "TIFSupport",
    "cast_supported_capabilities",
]
