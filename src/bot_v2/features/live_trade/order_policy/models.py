"""Dataclasses describing order policy capabilities and symbol policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .enums import OrderTypeSupport


@dataclass
class OrderCapability:
    """Order type capability definition."""

    order_type: str
    tif: str
    support_level: OrderTypeSupport
    min_quantity: Decimal | None = None
    max_quantity: Decimal | None = None
    quantity_increment: Decimal | None = None
    price_increment: Decimal | None = None
    post_only_supported: bool = True
    reduce_only_supported: bool = True
    bracket_supported: bool = False
    max_notional: Decimal | None = None
    rate_limit_per_minute: int = 60

    def to_dict(self) -> dict[str, Any]:
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
    environment: str
    capabilities: list[OrderCapability] = field(default_factory=list)
    min_order_size: Decimal = Decimal("0.001")
    max_order_size: Decimal | None = None
    size_increment: Decimal = Decimal("0.001")
    price_increment: Decimal = Decimal("0.01")
    max_position_size: Decimal | None = None
    max_daily_volume: Decimal | None = None
    trading_enabled: bool = True
    reduce_only_mode: bool = False
    requires_post_only: bool = False
    spread_threshold_bps: Decimal | None = None

    def get_capability(self, order_type: str, tif: str) -> OrderCapability | None:
        for cap in self.capabilities:
            if cap.order_type == order_type and cap.tif == tif:
                return cap
        return None

    def is_order_allowed(
        self,
        order_type: str,
        tif: str,
        quantity: Decimal,
        price: Decimal | None = None,
    ) -> tuple[bool, str]:
        capability = self.get_capability(order_type, tif)
        if not self.trading_enabled:
            return False, "Trading disabled for symbol"
        if self.reduce_only_mode and order_type not in {"MARKET", "LIMIT"}:
            return False, "Reduce-only mode: only market/limit orders allowed"
        if capability is None:
            return False, f"No capability for {order_type}/{tif}"
        if capability.support_level == OrderTypeSupport.UNSUPPORTED:
            return False, f"{order_type} with {tif} unsupported"
        if capability.support_level == OrderTypeSupport.GATED:
            return False, f"{order_type} with {tif} gated"
        if quantity < self.min_order_size:
            return False, f"Quantity {quantity} below minimum {self.min_order_size}"
        if self.max_order_size and quantity > self.max_order_size:
            return False, f"Quantity {quantity} exceeds maximum {self.max_order_size}"
        if capability.min_quantity and quantity < capability.min_quantity:
            return False, f"Quantity {quantity} below min {capability.min_quantity}"
        if capability.max_quantity and quantity > capability.max_quantity:
            return False, f"Quantity {quantity} exceeds max {capability.max_quantity}"
        if self.requires_post_only and tif != "GTC":
            return False, "Post-only mode requires GTC"
        if capability.quantity_increment:
            quantity_remainder = quantity % capability.quantity_increment
            if quantity_remainder != 0:
                return False, (
                    f"Quantity {quantity} not aligned to increment {capability.quantity_increment}"
                )
        if capability.price_increment and price is not None:
            price_remainder = price % capability.price_increment
            if price_remainder != 0:
                return False, f"Price {price} not aligned to increment {capability.price_increment}"
        if price and capability.max_notional:
            notional = quantity * price
            if notional > capability.max_notional:
                return False, f"Notional {notional} exceeds maximum {capability.max_notional}"
        return True, "Order allowed"

    def to_dict(self) -> dict[str, Any]:
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


__all__ = ["OrderCapability", "SymbolPolicy"]
