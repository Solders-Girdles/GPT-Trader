"""
Application configuration module.

This is the canonical location for BotConfig and related configuration types.
Part of the modern DI layer (app/).

Types:
    BotConfig: Main bot configuration dataclass
    BotRiskConfig: Risk/position sizing configuration
    MeanReversionConfig: Mean reversion strategy configuration
    StrategyType: Literal type for strategy selection

Constants:
    DEFAULT_SPOT_RISK_PATH: Default path for spot risk config
    DEFAULT_SPOT_SYMBOLS: Default symbols for spot trading
    TOP_VOLUME_BASES: Top volume base currencies

Usage:
    from gpt_trader.app.config import BotConfig, BotRiskConfig

    config = BotConfig.from_env()
    # or
    config = BotConfig.from_profile(Profile.DEV, dry_run=True)
"""

from gpt_trader.app.config.bot_config import (
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
    BotConfig,
    BotRiskConfig,
    MeanReversionConfig,
    StrategyType,
)

__all__ = [
    "BotConfig",
    "BotRiskConfig",
    "MeanReversionConfig",
    "StrategyType",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
]
