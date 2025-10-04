"""
Order Policy Matrix for Production Trading.

Manages exchange capability awareness, order type support,
and trading policy enforcement per symbol and environment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, TypedDict, cast

from bot_v2.features.live_trade.capability_registry import (
    CapabilityRegistry,
    OrderCapability,
    OrderTypeSupport,
)
from bot_v2.features.live_trade.order_recommender import OrderConfig, OrderRecommender
from bot_v2.features.live_trade.policy_validator import PolicyValidator
from bot_v2.features.live_trade.rate_limit_tracker import RateLimitTracker

logger = logging.getLogger(__name__)


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
        return CapabilityRegistry.find_capability(self.capabilities, order_type, tif)

    def is_order_allowed(
        self, order_type: str, tif: str, quantity: Decimal, price: Decimal | None = None
    ) -> tuple[bool, str]:
        """
        Check if order is allowed under current policy.

        Returns:
            Tuple of (allowed, reason)
        """
        capability = self.get_capability(order_type, tif)

        return PolicyValidator.validate_order(
            order_type=order_type,
            tif=tif,
            quantity=quantity,
            price=price,
            trading_enabled=self.trading_enabled,
            min_order_size=self.min_order_size,
            max_order_size=self.max_order_size,
            size_increment=self.size_increment,
            price_increment=self.price_increment,
            capability=capability,
            environment=self.environment,
        )

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


class SupportedOrderConfig(TypedDict):
    order_type: str
    tif: str
    post_only: bool
    reduce_only: bool


class OrderPolicyMatrix:
    """
    Production order policy matrix.

    Manages order type support, TIF capabilities, and trading
    policies across symbols and environments.
    """

    def __init__(
        self,
        environment: str = "sandbox",
        rate_limit_tracker: RateLimitTracker | None = None,
    ) -> None:
        self.environment = environment
        self._symbol_policies: dict[str, SymbolPolicy] = {}
        self._rate_limit_tracker = rate_limit_tracker or RateLimitTracker()

        logger.info(f"OrderPolicyMatrix initialized for {environment} environment")

    def add_symbol(
        self,
        symbol: str,
        capabilities: list[OrderCapability] | None = None,
        **policy_kwargs: Any,
    ) -> SymbolPolicy:
        """Add symbol with trading policy."""
        if capabilities is None:
            capabilities = CapabilityRegistry.get_coinbase_perp_capabilities()

        policy = SymbolPolicy(
            symbol=symbol, environment=self.environment, capabilities=capabilities, **policy_kwargs
        )

        self._symbol_policies[symbol] = policy

        logger.info(f"Added symbol policy: {symbol} ({len(capabilities)} capabilities)")
        return policy

    def get_symbol_policy(self, symbol: str) -> SymbolPolicy | None:
        """Get policy for symbol."""
        return self._symbol_policies.get(symbol)

    def validate_order(
        self,
        symbol: str,
        order_type: str,
        tif: str,
        quantity: Decimal,
        price: Decimal | None = None,
        post_only: bool = False,
        reduce_only: bool = False,
    ) -> tuple[bool, str]:
        """
        Validate order against policy matrix.

        Returns:
            Tuple of (allowed, reason)
        """
        # Check if symbol has policy
        policy = self.get_symbol_policy(symbol)
        if not policy:
            return False, f"No policy defined for symbol {symbol}"

        # Get capability for rate limit check
        capability = policy.get_capability(order_type, tif)
        if not capability:
            return False, f"No capability for {order_type}/{tif}"

        # Validate order using PolicyValidator
        allowed, reason = PolicyValidator.validate_order(
            order_type=order_type,
            tif=tif,
            quantity=quantity,
            price=price,
            trading_enabled=policy.trading_enabled,
            min_order_size=policy.min_order_size,
            max_order_size=policy.max_order_size,
            size_increment=policy.size_increment,
            price_increment=policy.price_increment,
            capability=capability,
            post_only=post_only,
            reduce_only=reduce_only,
            environment=self.environment,
        )
        if not allowed:
            return False, reason

        # Check rate limits
        if not self._check_rate_limit(symbol, capability.rate_limit_per_minute):
            return False, f"Rate limit exceeded ({capability.rate_limit_per_minute}/min)"

        return True, "Order validated"

    def get_supported_order_types(self, symbol: str) -> list[SupportedOrderConfig]:
        """Get supported order types for symbol."""
        policy = self.get_symbol_policy(symbol)
        if not policy:
            return []

        supported: list[SupportedOrderConfig] = []
        for capability in policy.capabilities:
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

    def recommend_order_config(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        urgency: str = "normal",  # "urgent", "normal", "patient"
        market_conditions: Mapping[str, float | int | str | bool] | None = None,
    ) -> OrderConfig:
        """
        Recommend order configuration based on conditions.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            urgency: Execution urgency
            market_conditions: Current market state

        Returns:
            Recommended order configuration
        """
        policy = self.get_symbol_policy(symbol)
        if not policy:
            return cast(OrderConfig, {"error": f"No policy for {symbol}"})

        # Get recommendation from OrderRecommender
        config = OrderRecommender.recommend_config(
            symbol_policy=policy,
            side=side,
            quantity=quantity,
            urgency=urgency,
            market_conditions=market_conditions,
        )

        # Validate recommended configuration
        allowed, reason = self.validate_order(
            symbol=symbol,
            order_type=config["order_type"],
            tif=config["tif"],
            quantity=quantity,
            post_only=config["post_only"],
            reduce_only=config["reduce_only"],
        )

        if not allowed:
            # Fallback to basic limit GTC
            config = cast(
                OrderConfig,
                {
                    "order_type": "LIMIT",
                    "tif": "GTC",
                    "post_only": False,
                    "reduce_only": False,
                    "use_market": False,
                    "fallback_reason": reason,
                },
            )

        return config

    def enable_gtd_orders(self, symbol: str) -> bool:
        """Enable GTD orders for symbol if they were gated."""
        policy = self.get_symbol_policy(symbol)
        if not policy:
            return False

        gtd_enabled = CapabilityRegistry.enable_gtd_orders(policy.capabilities)

        if gtd_enabled:
            logger.info(f"GTD orders enabled for {symbol}")

        return gtd_enabled

    def set_reduce_only_mode(self, symbol: str, enabled: bool = True) -> None:
        """Enable reduce-only mode for symbol."""
        policy = self.get_symbol_policy(symbol)
        if policy:
            policy.reduce_only_mode = enabled
            logger.info(f"Reduce-only mode {'enabled' if enabled else 'disabled'} for {symbol}")

    def get_policy_summary(self) -> dict[str, Any]:
        """Get summary of all symbol policies."""
        summary: dict[str, Any] = {
            "environment": self.environment,
            "symbols": len(self._symbol_policies),
            "policies": {},
        }

        for symbol, policy in self._symbol_policies.items():
            supported_types = [
                cap.order_type
                for cap in policy.capabilities
                if cap.support_level == OrderTypeSupport.SUPPORTED
            ]

            summary["policies"][symbol] = {
                "trading_enabled": policy.trading_enabled,
                "reduce_only_mode": policy.reduce_only_mode,
                "supported_order_types": list(set(supported_types)),
                "min_order_size": float(policy.min_order_size),
                "capabilities_count": len(policy.capabilities),
            }

        return summary

    def _check_rate_limit(self, symbol: str, limit_per_minute: int) -> bool:
        """Check if symbol is within rate limit."""
        return self._rate_limit_tracker.check_and_record(symbol, limit_per_minute)


def create_standard_policy_matrix(environment: str = "sandbox") -> OrderPolicyMatrix:
    """Create standard policy matrix with common perpetuals."""
    matrix = OrderPolicyMatrix(environment=environment)

    # Standard perpetuals symbols
    symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]

    for symbol in symbols:
        # Symbol-specific increments (would be fetched from exchange)
        increments = {
            "BTC-USD": {"size": Decimal("0.001"), "price": Decimal("1")},
            "ETH-USD": {"size": Decimal("0.01"), "price": Decimal("0.1")},
            "SOL-USD": {"size": Decimal("0.1"), "price": Decimal("0.001")},
            "XRP-USD": {"size": Decimal("1"), "price": Decimal("0.0001")},
        }

        increment = increments.get(symbol, {"size": Decimal("0.001"), "price": Decimal("0.01")})

        matrix.add_symbol(
            symbol=symbol,
            min_order_size=increment["size"],
            size_increment=increment["size"],
            price_increment=increment["price"],
            max_order_size=Decimal("1000"),  # Reasonable limit
            trading_enabled=True,
            reduce_only_mode=False,
            spread_threshold_bps=Decimal("20"),  # 2bps threshold for post-only
        )

    logger.info(f"Standard policy matrix created with {len(symbols)} symbols")
    return matrix


async def create_order_policy_matrix(environment: str = "sandbox") -> OrderPolicyMatrix:
    """Create and initialize order policy matrix."""
    matrix = create_standard_policy_matrix(environment=environment)
    logger.info(f"OrderPolicyMatrix created for {environment}")
    return matrix
