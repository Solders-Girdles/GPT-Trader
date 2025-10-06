"""Strategy registry for managing trading strategy initialization and retrieval.

This module provides utilities for creating and managing trading strategies based on
profile configuration (SPOT vs PERPS), with support for per-symbol strategies and
shared strategies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bot_v2.features.live_trade.profiles import SpotProfileService
from bot_v2.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    StrategyConfig,
)
from bot_v2.orchestration.configuration import Profile

if TYPE_CHECKING:
    from bot_v2.features.live_trade.risk_runtime import RiskManager
    from bot_v2.orchestration.configuration import BotConfig

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Manages strategy initialization and retrieval per symbol.

    This registry handles two different strategy patterns:
    1. SPOT profile: Per-symbol strategies with custom parameters
    2. PERPS profile: Single shared strategy for all symbols

    Example:
        >>> registry = StrategyRegistry(config, risk_manager)
        >>> registry.initialize()
        >>> strategy = registry.get_strategy("BTC-USD")
    """

    def __init__(
        self,
        config: BotConfig,
        risk_manager: RiskManager,
        spot_profile_service: SpotProfileService | None = None,
    ) -> None:
        """Initialize strategy registry.

        Args:
            config: Bot configuration
            risk_manager: Risk manager instance
            spot_profile_service: Service for loading spot profile rules (optional)
        """
        self.config = config
        self.risk_manager = risk_manager
        self.spot_profiles = spot_profile_service or SpotProfileService()
        self._symbol_strategies: dict[str, BaselinePerpsStrategy] = {}
        self._default_strategy: BaselinePerpsStrategy | None = None

    def initialize(self) -> None:
        """Initialize strategies based on profile configuration.

        For SPOT profile:
            - Creates per-symbol strategies with custom parameters from spot profiles
            - Loads short/long MA windows from spot profile rules
            - Applies position_fraction overrides with validation

        For PERPS profile:
            - Creates single shared strategy for all symbols
            - Applies derivatives settings (leverage, shorts)
            - Uses global position_fraction override
        """
        derivatives_enabled = self.config.derivatives_enabled

        if self.config.profile == Profile.SPOT:
            self._initialize_spot_strategies()
        else:
            self._initialize_perps_strategy(derivatives_enabled)

    def _initialize_spot_strategies(self) -> None:
        """Initialize per-symbol strategies for SPOT profile."""
        rules = self.spot_profiles.load(self.config.symbols or [])

        for symbol in self.config.symbols or []:
            rule = rules.get(symbol, {})
            short = int(rule.get("short_window", self.config.short_ma))
            long = int(rule.get("long_window", self.config.long_ma))

            strategy_kwargs = {
                "short_ma_period": short,
                "long_ma_period": long,
                "target_leverage": 1,
                "trailing_stop_pct": self.config.trailing_stop_pct,
                "enable_shorts": False,
            }

            # Apply position_fraction override
            fraction_override = rule.get("position_fraction")
            if fraction_override is None:
                fraction_override = self.config.perps_position_fraction

            if fraction_override is not None:
                try:
                    strategy_kwargs["position_fraction"] = float(fraction_override)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid position_fraction=%s for %s; using default",
                        fraction_override,
                        symbol,
                    )

            self._symbol_strategies[symbol] = BaselinePerpsStrategy(
                config=StrategyConfig(**strategy_kwargs),  # type: ignore[arg-type]
                risk_manager=self.risk_manager,
            )

    def _initialize_perps_strategy(self, derivatives_enabled: bool) -> None:
        """Initialize shared strategy for PERPS profile."""
        strategy_kwargs = {
            "short_ma_period": self.config.short_ma,
            "long_ma_period": self.config.long_ma,
            "target_leverage": self.config.target_leverage if derivatives_enabled else 1,
            "trailing_stop_pct": self.config.trailing_stop_pct,
            "enable_shorts": self.config.enable_shorts if derivatives_enabled else False,
        }

        fraction_override = self.config.perps_position_fraction
        if fraction_override is not None:
            try:
                strategy_kwargs["position_fraction"] = float(fraction_override)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid PERPS_POSITION_FRACTION=%s; using default",
                    fraction_override,
                )

        self._default_strategy = BaselinePerpsStrategy(
            config=StrategyConfig(**strategy_kwargs),  # type: ignore[arg-type]
            risk_manager=self.risk_manager,
        )

    def get_strategy(self, symbol: str) -> BaselinePerpsStrategy:
        """Get strategy for symbol.

        For SPOT profile:
            - Returns symbol-specific strategy if it exists
            - Creates and caches default strategy if symbol not found

        For PERPS profile:
            - Returns shared default strategy

        Args:
            symbol: Trading symbol

        Returns:
            Strategy instance for the symbol

        Example:
            >>> strategy = registry.get_strategy("BTC-USD")
        """
        if self.config.profile == Profile.SPOT:
            strat = self._symbol_strategies.get(symbol)
            if strat is None:
                # Lazy creation for missing symbols
                strat = BaselinePerpsStrategy(risk_manager=self.risk_manager)
                self._symbol_strategies[symbol] = strat
            return strat

        return self.default_strategy

    @property
    def default_strategy(self) -> BaselinePerpsStrategy:
        """Get default/shared strategy (for PERPS profile).

        Returns:
            Default strategy instance

        Raises:
            RuntimeError: If strategy not initialized
        """
        if self._default_strategy is None:
            raise RuntimeError("Default strategy not initialized. Call initialize() first.")
        return self._default_strategy

    @property
    def symbol_strategies(self) -> dict[str, BaselinePerpsStrategy]:
        """Get symbol-specific strategies (for SPOT profile).

        Returns:
            Dictionary mapping symbols to strategies
        """
        return self._symbol_strategies
