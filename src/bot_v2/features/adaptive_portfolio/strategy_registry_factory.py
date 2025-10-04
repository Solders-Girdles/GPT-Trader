"""Strategy registry factory - builds handler registry from configuration.

This module provides a factory for creating strategy handler registries based on
tier configuration. It maps strategy names to handler classes and handles dependency
injection and special cases (e.g., ML enhanced wrapping momentum).

Supported Strategy IDs:
    - "momentum": MomentumStrategyHandler - 5d/20d return momentum
    - "mean_reversion": MeanReversionStrategyHandler - z-score oversold detection
    - "trend_following": TrendFollowingStrategyHandler - MA alignment (10/30/50)
    - "ml_enhanced": MLEnhancedStrategyHandler - ML-boosted momentum signals

Custom Extensions:
    To add custom strategies:
    1. Implement StrategyHandler protocol (generate_signals method)
    2. Add to STRATEGY_HANDLER_MAP below
    3. Update tier config strategies list with new handler ID
"""

from typing import TYPE_CHECKING

from bot_v2.data_providers import DataProvider
from bot_v2.features.adaptive_portfolio.position_size_calculator import PositionSizeCalculator
from bot_v2.features.adaptive_portfolio.strategy_handlers import (
    MeanReversionStrategyHandler,
    MLEnhancedStrategyHandler,
    MomentumStrategyHandler,
    StrategyHandler,
    TrendFollowingStrategyHandler,
)
from bot_v2.features.adaptive_portfolio.types import TierConfig

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


class StrategyRegistryError(Exception):
    """Raised when strategy registry cannot be built from config."""

    pass


# Handler factory functions - each creates a handler instance
def _create_momentum_handler(
    data_provider: DataProvider,
    position_size_calculator: PositionSizeCalculator,
) -> MomentumStrategyHandler:
    """Create momentum strategy handler."""
    return MomentumStrategyHandler(data_provider, position_size_calculator)


def _create_mean_reversion_handler(
    data_provider: DataProvider,
    position_size_calculator: PositionSizeCalculator,
) -> MeanReversionStrategyHandler:
    """Create mean reversion strategy handler."""
    return MeanReversionStrategyHandler(data_provider, position_size_calculator)


def _create_trend_following_handler(
    data_provider: DataProvider,
    position_size_calculator: PositionSizeCalculator,
) -> TrendFollowingStrategyHandler:
    """Create trend following strategy handler."""
    return TrendFollowingStrategyHandler(data_provider, position_size_calculator)


# Strategy handler mapping - maps IDs to factory functions
STRATEGY_HANDLER_MAP: dict[
    str, "Callable[[DataProvider, PositionSizeCalculator], StrategyHandler]"
] = {
    "momentum": _create_momentum_handler,
    "mean_reversion": _create_mean_reversion_handler,
    "trend_following": _create_trend_following_handler,
}


def build_strategy_registry(
    tier_config: TierConfig,
    data_provider: DataProvider,
    position_size_calculator: PositionSizeCalculator,
) -> dict[str, StrategyHandler]:
    """Build strategy handler registry from tier configuration.

    Creates handler instances for all strategies declared in tier_config.strategies,
    wiring dependencies and handling special cases like ML enhanced.

    Args:
        tier_config: Tier configuration with strategies list
        data_provider: Data provider for handlers to use
        position_size_calculator: Position sizing calculator for handlers

    Returns:
        Dictionary mapping strategy names to handler instances

    Raises:
        StrategyRegistryError: If unknown strategy ID found or ml_enhanced
                               requested without momentum

    Example:
        >>> registry = build_strategy_registry(
        ...     tier_config,
        ...     data_provider,
        ...     position_size_calculator
        ... )
        >>> signals = registry["momentum"].generate_signals(...)
    """
    registry: dict[str, StrategyHandler] = {}
    unknown_strategies: list[str] = []

    # Build registry for all declared strategies
    for strategy_name in tier_config.strategies:
        if strategy_name == "ml_enhanced":
            # Special case: ml_enhanced requires momentum to wrap
            continue

        factory = STRATEGY_HANDLER_MAP.get(strategy_name)
        if factory is None:
            unknown_strategies.append(strategy_name)
            continue

        registry[strategy_name] = factory(data_provider, position_size_calculator)

    # Handle ml_enhanced special case
    if "ml_enhanced" in tier_config.strategies:
        if "momentum" not in registry:
            raise StrategyRegistryError(
                "ml_enhanced strategy requires momentum handler to be enabled. "
                f"Tier '{tier_config.name}' strategies: {tier_config.strategies}"
            )

        # Wrap momentum handler with ML enhancement
        momentum_handler = registry["momentum"]
        registry["ml_enhanced"] = MLEnhancedStrategyHandler(momentum_handler)

    # Report unknown strategies
    if unknown_strategies:
        raise StrategyRegistryError(
            f"Unknown strategy IDs in tier '{tier_config.name}': {unknown_strategies}. "
            f"Supported strategies: {sorted(STRATEGY_HANDLER_MAP.keys()) + ['ml_enhanced']}"
        )

    return registry


def get_supported_strategies() -> list[str]:
    """Get list of all supported strategy IDs.

    Returns:
        List of strategy IDs that can be used in tier config
    """
    return sorted(STRATEGY_HANDLER_MAP.keys()) + ["ml_enhanced"]
