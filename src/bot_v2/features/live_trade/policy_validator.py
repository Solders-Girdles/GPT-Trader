"""
Policy Validator for Order Validation.

Validates orders against symbol policies, checking quantity/price alignment,
notional limits, and capability-specific requirements.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from bot_v2.features.live_trade.capability_registry import OrderCapability, OrderTypeSupport

logger = logging.getLogger(__name__)


class PolicyValidator:
    """
    Validates orders against trading policies.

    Stateless validator that checks orders against symbol policies
    for compliance with trading rules, limits, and increments.
    """

    @staticmethod
    def validate_trading_enabled(trading_enabled: bool) -> tuple[bool, str]:
        """
        Check if trading is enabled.

        Args:
            trading_enabled: Whether trading is enabled for symbol

        Returns:
            Tuple of (allowed, reason)
        """
        if not trading_enabled:
            return False, "Trading disabled for symbol"
        return True, ""

    @staticmethod
    def validate_capability_support(
        capability: OrderCapability | None, order_type: str, tif: str
    ) -> tuple[bool, str]:
        """
        Validate capability exists and is supported.

        Args:
            capability: Capability to validate
            order_type: Requested order type
            tif: Requested time-in-force

        Returns:
            Tuple of (allowed, reason)
        """
        if not capability:
            return False, f"Order type {order_type} with TIF {tif} not supported"

        if capability.support_level == OrderTypeSupport.UNSUPPORTED:
            return False, f"Order type {order_type} unsupported"

        if capability.support_level == OrderTypeSupport.GATED:
            return False, f"Order type {order_type} currently gated"

        return True, ""

    @staticmethod
    def validate_quantity_limits(
        quantity: Decimal,
        min_order_size: Decimal,
        max_order_size: Decimal | None,
        capability: OrderCapability | None,
    ) -> tuple[bool, str]:
        """
        Validate quantity against min/max limits.

        Args:
            quantity: Order quantity
            min_order_size: Symbol minimum order size
            max_order_size: Symbol maximum order size (optional)
            capability: Order capability with capability-specific limits

        Returns:
            Tuple of (allowed, reason)
        """
        # Check symbol min/max
        if quantity < min_order_size:
            return False, f"Quantity {quantity} below minimum {min_order_size}"

        if max_order_size and quantity > max_order_size:
            return False, f"Quantity {quantity} exceeds maximum {max_order_size}"

        # Check capability-specific limits
        if capability:
            if capability.min_quantity and quantity < capability.min_quantity:
                return (
                    False,
                    f"Quantity {quantity} below capability minimum {capability.min_quantity}",
                )

            if capability.max_quantity and quantity > capability.max_quantity:
                return (
                    False,
                    f"Quantity {quantity} exceeds capability maximum {capability.max_quantity}",
                )

        return True, ""

    @staticmethod
    def validate_quantity_increment(quantity: Decimal, size_increment: Decimal) -> tuple[bool, str]:
        """
        Validate quantity aligns to increment.

        Args:
            quantity: Order quantity
            size_increment: Required size increment

        Returns:
            Tuple of (allowed, reason)
        """
        remainder = quantity % size_increment
        if remainder != 0:
            return False, f"Quantity {quantity} not aligned to increment {size_increment}"

        return True, ""

    @staticmethod
    def validate_price_increment(
        price: Decimal | None, price_increment: Decimal
    ) -> tuple[bool, str]:
        """
        Validate price aligns to increment.

        Args:
            price: Order price (None for market orders)
            price_increment: Required price increment

        Returns:
            Tuple of (allowed, reason)
        """
        if price is None:
            return True, ""  # Market orders don't need price validation

        price_remainder = price % price_increment
        if price_remainder != 0:
            return False, f"Price {price} not aligned to increment {price_increment}"

        return True, ""

    @staticmethod
    def validate_notional_limit(
        quantity: Decimal, price: Decimal | None, capability: OrderCapability | None
    ) -> tuple[bool, str]:
        """
        Validate notional (quantity * price) against limit.

        Args:
            quantity: Order quantity
            price: Order price (None for market orders)
            capability: Order capability with max_notional limit

        Returns:
            Tuple of (allowed, reason)
        """
        if price is None or capability is None or capability.max_notional is None:
            return True, ""  # No notional check needed

        notional = quantity * price
        if notional > capability.max_notional:
            return False, f"Notional {notional} exceeds maximum {capability.max_notional}"

        return True, ""

    @staticmethod
    def validate_post_only_support(
        post_only: bool, capability: OrderCapability | None
    ) -> tuple[bool, str]:
        """
        Validate post-only flag is supported.

        Args:
            post_only: Whether post-only is requested
            capability: Order capability

        Returns:
            Tuple of (allowed, reason)
        """
        if not post_only:
            return True, ""  # Not requesting post-only, no check needed

        if capability and not capability.post_only_supported:
            return False, "Post-only not supported for this order type"

        return True, ""

    @staticmethod
    def validate_reduce_only_support(
        reduce_only: bool, capability: OrderCapability | None
    ) -> tuple[bool, str]:
        """
        Validate reduce-only flag is supported.

        Args:
            reduce_only: Whether reduce-only is requested
            capability: Order capability

        Returns:
            Tuple of (allowed, reason)
        """
        if not reduce_only:
            return True, ""  # Not requesting reduce-only, no check needed

        if capability and not capability.reduce_only_supported:
            return False, "Reduce-only not supported for this order type"

        return True, ""

    @staticmethod
    def validate_environment_rules(environment: str, order_type: str, tif: str) -> tuple[bool, str]:
        """
        Validate environment-specific rules.

        Args:
            environment: Trading environment ("paper", "sandbox", "live")
            order_type: Order type
            tif: Time-in-force

        Returns:
            Tuple of (allowed, reason)
        """
        # Paper trading restrictions
        if environment == "paper":
            if order_type in ["STOP", "STOP_LIMIT"] and tif == "GTD":
                return False, "GTD stop orders not allowed in paper trading"

        return True, ""

    @staticmethod
    def validate_order(
        order_type: str,
        tif: str,
        quantity: Decimal,
        price: Decimal | None,
        trading_enabled: bool,
        min_order_size: Decimal,
        max_order_size: Decimal | None,
        size_increment: Decimal,
        price_increment: Decimal,
        capability: OrderCapability | None,
        post_only: bool = False,
        reduce_only: bool = False,
        environment: str = "sandbox",
    ) -> tuple[bool, str]:
        """
        Validate order against all policy rules.

        Performs complete validation pipeline checking all constraints.

        Args:
            order_type: Order type (e.g., "LIMIT", "MARKET")
            tif: Time-in-force (e.g., "GTC", "IOC")
            quantity: Order quantity
            price: Order price (None for market orders)
            trading_enabled: Whether trading is enabled
            min_order_size: Minimum order size
            max_order_size: Maximum order size (optional)
            size_increment: Required size increment
            price_increment: Required price increment
            capability: Order capability
            post_only: Post-only flag
            reduce_only: Reduce-only flag
            environment: Trading environment

        Returns:
            Tuple of (allowed, reason)
        """
        # 1. Trading enabled check
        allowed, reason = PolicyValidator.validate_trading_enabled(trading_enabled)
        if not allowed:
            return False, reason

        # 2. Capability support check
        allowed, reason = PolicyValidator.validate_capability_support(capability, order_type, tif)
        if not allowed:
            return False, reason

        # 3. Quantity limits
        allowed, reason = PolicyValidator.validate_quantity_limits(
            quantity, min_order_size, max_order_size, capability
        )
        if not allowed:
            return False, reason

        # 4. Quantity increment
        allowed, reason = PolicyValidator.validate_quantity_increment(quantity, size_increment)
        if not allowed:
            return False, reason

        # 5. Price increment
        allowed, reason = PolicyValidator.validate_price_increment(price, price_increment)
        if not allowed:
            return False, reason

        # 6. Notional limit
        allowed, reason = PolicyValidator.validate_notional_limit(quantity, price, capability)
        if not allowed:
            return False, reason

        # 7. Post-only support
        allowed, reason = PolicyValidator.validate_post_only_support(post_only, capability)
        if not allowed:
            return False, reason

        # 8. Reduce-only support
        allowed, reason = PolicyValidator.validate_reduce_only_support(reduce_only, capability)
        if not allowed:
            return False, reason

        # 9. Environment-specific rules
        allowed, reason = PolicyValidator.validate_environment_rules(environment, order_type, tif)
        if not allowed:
            return False, reason

        return True, "Order allowed"
