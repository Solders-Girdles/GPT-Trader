"""Order policy matrix that coordinates symbol capabilities and limits."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, cast

from bot_v2.utilities.logging_patterns import get_logger

from .enums import OrderTypeSupport
from .models import OrderCapability, SymbolPolicy
from .types import OrderConfig, SupportedOrderConfig

logger = get_logger(__name__, component="live_trade_policy")


class OrderPolicyMatrix:
    """Production order policy matrix."""

    COINBASE_PERP_CAPABILITIES = [
        OrderCapability("MARKET", "IOC", OrderTypeSupport.SUPPORTED, post_only_supported=False),
        OrderCapability("LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
        OrderCapability("LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
        OrderCapability("LIMIT", "FOK", OrderTypeSupport.SUPPORTED),
        OrderCapability("STOP", "GTC", OrderTypeSupport.SUPPORTED),
        OrderCapability("STOP", "IOC", OrderTypeSupport.SUPPORTED),
        OrderCapability("STOP_LIMIT", "GTC", OrderTypeSupport.SUPPORTED),
        OrderCapability("STOP_LIMIT", "IOC", OrderTypeSupport.SUPPORTED),
        OrderCapability("LIMIT", "GTD", OrderTypeSupport.GATED),
        OrderCapability("STOP", "GTD", OrderTypeSupport.GATED),
        OrderCapability("STOP_LIMIT", "GTD", OrderTypeSupport.GATED),
    ]

    def __init__(self, environment: str = "sandbox") -> None:
        self.environment = environment
        self._symbol_policies: dict[str, SymbolPolicy] = {}
        self._rate_limits: dict[str, list[datetime]] = {}
        logger.info(f"OrderPolicyMatrix initialized for {environment} environment")

    def add_symbol(
        self,
        symbol: str,
        capabilities: list[OrderCapability] | None = None,
        **policy_kwargs: Any,
    ) -> SymbolPolicy:
        if capabilities is None:
            capabilities = self.COINBASE_PERP_CAPABILITIES.copy()

        policy = SymbolPolicy(
            symbol=symbol,
            environment=self.environment,
            capabilities=capabilities,
            **policy_kwargs,
        )
        self._symbol_policies[symbol] = policy
        logger.info(f"Added symbol policy: {symbol} ({len(capabilities)} capabilities)")
        return policy

    def get_symbol_policy(self, symbol: str) -> SymbolPolicy | None:
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
        policy = self.get_symbol_policy(symbol)
        if not policy:
            return False, f"No policy defined for symbol {symbol}"

        allowed, reason = policy.is_order_allowed(order_type, tif, quantity, price)
        if not allowed:
            return False, reason

        capability = policy.get_capability(order_type, tif)
        if not capability:
            return False, f"No capability for {order_type}/{tif}"
        if post_only and not capability.post_only_supported:
            return False, "Post-only not supported for this order type"
        if reduce_only and not capability.reduce_only_supported:
            return False, "Reduce-only not supported for this order type"
        if not self._check_rate_limit(symbol, capability.rate_limit_per_minute):
            return False, f"Rate limit exceeded ({capability.rate_limit_per_minute}/min)"
        if self.environment == "paper" and order_type in {"STOP", "STOP_LIMIT"} and tif == "GTD":
            return False, "GTD stop orders not allowed in paper trading"
        return True, "Order validated"

    def get_supported_order_types(self, symbol: str) -> list[SupportedOrderConfig]:
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
        urgency: str = "normal",
        market_conditions: Mapping[str, float | int | str | bool] | None = None,
    ) -> OrderConfig:
        policy = self.get_symbol_policy(symbol)
        if not policy:
            return cast(OrderConfig, {"error": f"No policy for {symbol}"})

        config: OrderConfig = {
            "order_type": "LIMIT",
            "tif": "GTC",
            "post_only": False,
            "reduce_only": False,
            "use_market": False,
        }

        if urgency == "urgent":
            if market_conditions and market_conditions.get("liquidity_condition") in {
                "good",
                "excellent",
            }:
                config.update({"order_type": "MARKET", "tif": "IOC", "use_market": True})
            else:
                config["tif"] = "IOC"
        elif urgency == "patient":
            config["post_only"] = True

        if market_conditions:
            spread_raw = market_conditions.get("spread_bps", 0)
            try:
                spread_bps = Decimal(str(spread_raw))
            except (InvalidOperation, ValueError, TypeError):
                spread_bps = Decimal("0")
            if policy.spread_threshold_bps and spread_bps > policy.spread_threshold_bps:
                config["post_only"] = True
                config["order_type"] = "LIMIT"
                config["use_market"] = False

            volatility_raw = market_conditions.get("volatility_percentile", 0)
            try:
                volatility_percentile = float(volatility_raw)
            except (TypeError, ValueError):
                volatility_percentile = 0.0
            if volatility_percentile > 90:
                config["tif"] = "IOC"

        allowed, reason = self.validate_order(
            symbol=symbol,
            order_type=config["order_type"],
            tif=config["tif"],
            quantity=quantity,
            post_only=config["post_only"],
            reduce_only=config["reduce_only"],
        )

        if not allowed:
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
        policy = self.get_symbol_policy(symbol)
        if policy:
            policy.reduce_only_mode = enabled
            logger.info(f"Reduce-only mode {'enabled' if enabled else 'disabled'} for {symbol}")

    def get_policy_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "environment": self.environment,
            "symbols": len(self._symbol_policies),
            "policies": {},
        }

        for symbol, policy in self._symbol_policies.items():
            supported_types = {
                cap.order_type
                for cap in policy.capabilities
                if cap.support_level == OrderTypeSupport.SUPPORTED
            }
            summary["policies"][symbol] = {
                "trading_enabled": policy.trading_enabled,
                "reduce_only_mode": policy.reduce_only_mode,
                "supported_order_types": list(supported_types),
                "min_order_size": float(policy.min_order_size),
                "capabilities_count": len(policy.capabilities),
            }
        return summary

    async def get_capabilities(self, symbol: str) -> dict[str, bool]:
        return {"limit": True, "stop_limit": True, "gtd_gated": True}

    def _check_rate_limit(self, symbol: str, limit_per_minute: int) -> bool:
        now = datetime.now()
        cutoff = now - timedelta(minutes=1)

        if symbol not in self._rate_limits:
            self._rate_limits[symbol] = []

        self._rate_limits[symbol] = [ts for ts in self._rate_limits[symbol] if ts >= cutoff]
        if len(self._rate_limits[symbol]) >= limit_per_minute:
            return False

        self._rate_limits[symbol].append(now)
        return True


__all__ = ["OrderPolicyMatrix"]
