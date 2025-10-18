"""
Order Policy Matrix for Production Trading.

Manages exchange capability awareness, order type support,
and trading policy enforcement per symbol and environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, TypedDict, cast

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_trade_policy")


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

        # Check capability
        capability = self.get_capability(order_type, tif)
        if not capability:
            return False, f"Order type {order_type} with TIF {tif} not supported"

        if capability.support_level == OrderTypeSupport.UNSUPPORTED:
            return False, f"Order type {order_type} unsupported"

        if capability.support_level == OrderTypeSupport.GATED:
            return False, f"Order type {order_type} currently gated"

        # Check quantity limits
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

        # Check quantity increment
        remainder = quantity % self.size_increment
        if remainder != 0:
            return False, f"Quantity {quantity} not aligned to increment {self.size_increment}"

        # Check price increment
        if price and self.price_increment:
            price_remainder = price % self.price_increment
            if price_remainder != 0:
                return False, f"Price {price} not aligned to increment {self.price_increment}"

        # Check notional limits
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


class OrderPolicyMatrix:
    """
    Production order policy matrix.

    Manages order type support, TIF capabilities, and trading
    policies across symbols and environments.
    """

    # Standard Coinbase perpetuals capabilities
    COINBASE_PERP_CAPABILITIES = [
        # Market orders
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
        # GTD orders (gated until proven)
        OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
        OrderCapability("STOP", "GTD", OrderTypeSupport.GATED),
        OrderCapability("STOP_LIMIT", "GTD", OrderTypeSupport.GATED),
    ]

    def __init__(self, environment: str = "sandbox") -> None:
        self.environment = environment
        self._symbol_policies: dict[str, SymbolPolicy] = {}
        self._rate_limits: dict[str, list[datetime]] = {}  # symbol -> request timestamps

        logger.info(f"OrderPolicyMatrix initialized for {environment} environment")

    def add_symbol(
        self,
        symbol: str,
        capabilities: list[OrderCapability] | None = None,
        **policy_kwargs: Any,
    ) -> SymbolPolicy:
        """Add symbol with trading policy."""
        if capabilities is None:
            capabilities = self.COINBASE_PERP_CAPABILITIES.copy()

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

        # Check basic policy validation
        allowed, reason = policy.is_order_allowed(order_type, tif, quantity, price)
        if not allowed:
            return False, reason

        # Check capability-specific flags
        capability = policy.get_capability(order_type, tif)
        if not capability:
            return False, f"No capability for {order_type}/{tif}"

        if post_only and not capability.post_only_supported:
            return False, "Post-only not supported for this order type"

        if reduce_only and not capability.reduce_only_supported:
            return False, "Reduce-only not supported for this order type"

        # Check rate limits
        if not self._check_rate_limit(symbol, capability.rate_limit_per_minute):
            return False, f"Rate limit exceeded ({capability.rate_limit_per_minute}/min)"

        # Additional environment-specific checks
        if self.environment == "paper":
            # Paper trading might have additional restrictions
            if order_type in ["STOP", "STOP_LIMIT"] and tif == "GTD":
                return False, "GTD stop orders not allowed in paper trading"

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

        # Default configuration
        config: OrderConfig = {
            "order_type": "LIMIT",
            "tif": "GTC",
            "post_only": False,
            "reduce_only": False,
            "use_market": False,
        }

        # Adjust based on urgency
        if urgency == "urgent":
            # Prefer immediate execution
            if market_conditions and market_conditions.get("liquidity_condition") in [
                "good",
                "excellent",
            ]:
                config["order_type"] = "MARKET"
                config["tif"] = "IOC"
                config["use_market"] = True
            else:
                config["tif"] = "IOC"  # Limit IOC for poor liquidity
        elif urgency == "patient":
            # Prefer maker execution
            config["post_only"] = True

        # Adjust based on market conditions
        if market_conditions:
            spread_raw = market_conditions.get("spread_bps", 0)
            try:
                spread_bps = Decimal(str(spread_raw))
            except (InvalidOperation, ValueError, TypeError):
                spread_bps = Decimal("0")

            # Force post-only if spread is wide
            if policy.spread_threshold_bps and spread_bps > policy.spread_threshold_bps:
                config["post_only"] = True
                config["order_type"] = "LIMIT"
                config["use_market"] = False

            # Use IOC in volatile conditions
            volatility_raw = market_conditions.get("volatility_percentile", 0)
            try:
                volatility_percentile = float(volatility_raw)
            except (TypeError, ValueError):
                volatility_percentile = 0.0

            if volatility_percentile > 90:
                config["tif"] = "IOC"

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

        gtd_enabled = False
        for capability in policy.capabilities:
            if capability.tif == "GTD" and capability.support_level == OrderTypeSupport.GATED:
                capability.support_level = OrderTypeSupport.SUPPORTED
                gtd_enabled = True

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

    async def get_capabilities(self, symbol: str) -> dict[str, bool]:
        """Placeholder for get_capabilities."""
        return {"limit": True, "stop_limit": True, "gtd_gated": True}

    def _check_rate_limit(self, symbol: str, limit_per_minute: int) -> bool:
        """Check if symbol is within rate limit."""
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        # Initialize if needed
        if symbol not in self._rate_limits:
            self._rate_limits[symbol] = []

        # Clean old entries
        self._rate_limits[symbol] = [ts for ts in self._rate_limits[symbol] if ts >= cutoff]

        # Check limit
        if len(self._rate_limits[symbol]) >= limit_per_minute:
            return False

        # Add current request
        self._rate_limits[symbol].append(now)
        return True


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
