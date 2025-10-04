"""
Order Recommender for Configuration Selection.

Recommends order configurations (type, TIF, flags) based on
execution urgency and market conditions.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from bot_v2.features.live_trade.order_policy import SymbolPolicy

logger = logging.getLogger(__name__)


class OrderConfig(TypedDict, total=False):
    order_type: str
    tif: str
    post_only: bool
    reduce_only: bool
    use_market: bool
    fallback_reason: str
    error: str


class OrderRecommender:
    """
    Recommends order configurations based on market conditions.

    Stateless utility that analyzes urgency and market conditions
    to recommend optimal order type, TIF, and execution flags.
    """

    @staticmethod
    def recommend_config(
        symbol_policy: SymbolPolicy,
        side: str,
        quantity: Decimal,
        urgency: str = "normal",  # "urgent", "normal", "patient"
        market_conditions: Mapping[str, float | int | str | bool] | None = None,
    ) -> OrderConfig:
        """
        Recommend order configuration based on conditions.

        Args:
            symbol_policy: Symbol trading policy
            side: 'buy' or 'sell'
            quantity: Order quantity
            urgency: Execution urgency ("urgent", "normal", "patient")
            market_conditions: Current market state (spread_bps, volatility_percentile, liquidity_condition)

        Returns:
            Recommended order configuration
        """
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
            config = OrderRecommender._apply_urgent_urgency(config, market_conditions)
        elif urgency == "patient":
            config = OrderRecommender._apply_patient_urgency(config)

        # Adjust based on market conditions
        if market_conditions:
            config = OrderRecommender._apply_market_conditions(
                config, symbol_policy, market_conditions
            )

        return config

    @staticmethod
    def _apply_urgent_urgency(
        config: OrderConfig, market_conditions: Mapping[str, float | int | str | bool] | None
    ) -> OrderConfig:
        """
        Apply urgent urgency adjustments.

        Args:
            config: Current configuration
            market_conditions: Market state

        Returns:
            Updated configuration
        """
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

        return config

    @staticmethod
    def _apply_patient_urgency(config: OrderConfig) -> OrderConfig:
        """
        Apply patient urgency adjustments.

        Args:
            config: Current configuration

        Returns:
            Updated configuration
        """
        # Prefer maker execution
        config["post_only"] = True
        return config

    @staticmethod
    def _apply_market_conditions(
        config: OrderConfig,
        symbol_policy: SymbolPolicy,
        market_conditions: Mapping[str, float | int | str | bool],
    ) -> OrderConfig:
        """
        Apply market condition adjustments.

        Args:
            config: Current configuration
            symbol_policy: Symbol policy with thresholds
            market_conditions: Market state

        Returns:
            Updated configuration
        """
        # Handle spread conditions
        spread_raw = market_conditions.get("spread_bps", 0)
        try:
            spread_bps = Decimal(str(spread_raw))
        except (InvalidOperation, ValueError, TypeError):
            spread_bps = Decimal("0")

        # Force post-only if spread is wide
        if symbol_policy.spread_threshold_bps and spread_bps > symbol_policy.spread_threshold_bps:
            config["post_only"] = True
            config["order_type"] = "LIMIT"
            config["use_market"] = False

        # Handle volatility conditions
        volatility_raw = market_conditions.get("volatility_percentile", 0)
        try:
            volatility_percentile = float(volatility_raw)
        except (TypeError, ValueError):
            volatility_percentile = 0.0

        # Use IOC in volatile conditions
        if volatility_percentile > 90:
            config["tif"] = "IOC"

        return config
