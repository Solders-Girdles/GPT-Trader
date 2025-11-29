"""
Strategy factory for creating trading strategy instances.

This module provides a factory function that creates the appropriate
strategy based on the configuration, enabling runtime strategy selection.
"""

from typing import TYPE_CHECKING

from gpt_trader.features.live_trade.interfaces import TradingStrategy
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.orchestration.configuration.bot_config.bot_config import BotConfig

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

    else:
        raise ValueError(
            f"Unknown strategy_type: {strategy_type!r}. "
            f"Valid options: 'baseline', 'mean_reversion'"
        )
