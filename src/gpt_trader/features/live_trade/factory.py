"""
Strategy factory for creating trading strategy instances.

This module provides a factory function that creates the appropriate
strategy based on the configuration, enabling runtime strategy selection.
"""

from typing import TYPE_CHECKING

from gpt_trader.features.live_trade.interfaces import TradingStrategy
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.app.config import BotConfig

logger = get_logger(__name__, component="strategy_factory")


def create_strategy(config: "BotConfig") -> TradingStrategy:
    """Create a trading strategy based on configuration.

    Args:
        config: Bot configuration containing strategy_type and strategy-specific params

    Returns:
        A TradingStrategy instance ready for use by TradingEngine

    Raises:
        ValueError: If strategy_type is not recognized
    """
    strategy_type = config.strategy_type

    if strategy_type == "baseline":
        from gpt_trader.features.live_trade.strategies.perps_baseline import (
            BaselinePerpsStrategy,
        )

        logger.info("Creating BaselinePerpsStrategy (RSI + MA crossover)")
        return BaselinePerpsStrategy(config=config.strategy)

    elif strategy_type == "mean_reversion":
        from gpt_trader.features.live_trade.strategies.mean_reversion import (
            MeanReversionStrategy,
        )

        logger.info("Creating MeanReversionStrategy (Z-Score based)")
        return MeanReversionStrategy(config=config.mean_reversion)

    elif strategy_type == "ensemble":
        from gpt_trader.features.live_trade.strategies.ensemble import (
            EnsembleStrategy,
            EnsembleStrategyConfig,
        )

        logger.info("Creating EnsembleStrategy (Signal Ensemble Architecture)")

        # Use provided ensemble_config or default
        ensemble_config = config.ensemble_config
        if ensemble_config is None or not isinstance(ensemble_config, EnsembleStrategyConfig):
            # Try to parse from dict if it's a dict
            if isinstance(ensemble_config, dict):
                # This requires EnsembleStrategyConfig to be constructible from dict
                # For now, we'll just use default if parsing fails or implement a helper
                # But since we haven't implemented from_dict for EnsembleStrategyConfig,
                # let's assume it's passed correctly or use default.
                # Actually, let's try to unpack if it matches fields
                try:
                    ensemble_config = EnsembleStrategyConfig(**ensemble_config)
                except Exception:
                    logger.warning("Failed to parse ensemble_config dict, using default")
                    ensemble_config = EnsembleStrategyConfig()
            else:
                ensemble_config = EnsembleStrategyConfig()

        return EnsembleStrategy(config=ensemble_config)

    else:
        raise ValueError(
            f"Unknown strategy_type: {strategy_type!r}. "
            f"Valid options: 'baseline', 'mean_reversion', 'ensemble'"
        )
