"""
Capability Registry for Order Type Support.

Manages exchange capability definitions, order type support matrices,
and TIF (time-in-force) availability across different exchanges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class OrderTypeSupport(Enum):
    """Order type support levels."""

    SUPPORTED = "supported"
    GATED = "gated"  # Available but gated
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


class CapabilityRegistry:
    """
    Registry for exchange order capabilities.

    Provides default capability sets for different exchanges and
    utilities for capability lookup and modification.
    """

    @staticmethod
    def get_coinbase_perp_capabilities() -> list[OrderCapability]:
        """
        Get default Coinbase perpetuals capabilities.

        Returns:
            List of supported order type/TIF combinations for Coinbase perps
        """
        return [
            # Market orders (IOC only)
            OrderCapability("MARKET", "IOC", OrderTypeSupport.SUPPORTED, post_only_supported=False),
            # Limit orders
            OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
            OrderCapability("LIMIT", "FOK", OrderTypeSupport.SUPPORTED),
            # Stop orders
            OrderCapability("STOP", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("STOP", "IOC", OrderTypeSupport.SUPPORTED),
            # Stop-limit orders
            OrderCapability("STOP_LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
            OrderCapability("STOP_LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
            # GTD orders (gated until proven stable)
            OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
            OrderCapability("STOP", "GTD", OrderTypeSupport.GATED),
            OrderCapability("STOP_LIMIT", "GTD", OrderTypeSupport.GATED),
        ]

    @staticmethod
    def find_capability(
        capabilities: list[OrderCapability], order_type: str, tif: str
    ) -> OrderCapability | None:
        """
        Find capability matching order type and TIF.

        Args:
            capabilities: List of available capabilities
            order_type: Order type (e.g., "LIMIT", "MARKET", "STOP")
            tif: Time-in-force (e.g., "GTC", "IOC", "FOK", "GTD")

        Returns:
            Matching capability or None
        """
        for capability in capabilities:
            if capability.order_type == order_type and capability.tif == tif:
                return capability
        return None

    @staticmethod
    def enable_gtd_orders(capabilities: list[OrderCapability]) -> bool:
        """
        Enable GTD orders by changing GATED → SUPPORTED.

        Modifies capabilities in-place.

        Args:
            capabilities: List of capabilities to modify

        Returns:
            True if any GTD orders were enabled, False otherwise
        """
        enabled_count = 0

        for capability in capabilities:
            if capability.tif == "GTD" and capability.support_level == OrderTypeSupport.GATED:
                capability.support_level = OrderTypeSupport.SUPPORTED
                enabled_count += 1

        if enabled_count > 0:
            logger.info(f"Enabled {enabled_count} GTD order capabilities")
            return True

        return False

    @staticmethod
    def filter_supported(capabilities: list[OrderCapability]) -> list[OrderCapability]:
        """
        Filter to only SUPPORTED capabilities.

        Args:
            capabilities: List of capabilities

        Returns:
            Only capabilities with SUPPORTED status
        """
        return [cap for cap in capabilities if cap.support_level == OrderTypeSupport.SUPPORTED]

    @staticmethod
    def get_supported_order_types(capabilities: list[OrderCapability]) -> set[str]:
        """
        Get unique set of supported order types.

        Args:
            capabilities: List of capabilities

        Returns:
            Set of order type strings (e.g., {"LIMIT", "MARKET", "STOP"})
        """
        return {
            cap.order_type
            for cap in capabilities
            if cap.support_level == OrderTypeSupport.SUPPORTED
        }

    @staticmethod
    def validate_capability(capability: OrderCapability) -> tuple[bool, str]:
        """
        Validate capability definition.

        Args:
            capability: Capability to validate

        Returns:
            Tuple of (valid, error_message)
        """
        # Check required fields
        if not capability.order_type:
            return False, "order_type is required"

        if not capability.tif:
            return False, "tif is required"

        # Check quantity limits consistency
        if capability.min_quantity and capability.max_quantity:
            if capability.min_quantity > capability.max_quantity:
                return False, "min_quantity cannot exceed max_quantity"

        # Check increment positivity
        if capability.quantity_increment is not None and capability.quantity_increment <= 0:
            return False, "quantity_increment must be positive"

        if capability.price_increment is not None and capability.price_increment <= 0:
            return False, "price_increment must be positive"

        # Check rate limit
        if capability.rate_limit_per_minute < 1:
            return False, "rate_limit_per_minute must be at least 1"

        return True, ""

    @staticmethod
    def count_by_support_level(
        capabilities: list[OrderCapability],
    ) -> dict[OrderTypeSupport, int]:
        """
        Count capabilities by support level.

        Args:
            capabilities: List of capabilities

        Returns:
            Dict mapping support level to count
        """
        counts: dict[OrderTypeSupport, int] = {
            OrderTypeSupport.SUPPORTED: 0,
            OrderTypeSupport.GATED: 0,
            OrderTypeSupport.UNSUPPORTED: 0,
        }

        for capability in capabilities:
            counts[capability.support_level] += 1

        return counts
