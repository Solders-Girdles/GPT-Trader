"""
Application configuration module.

This is the canonical location for BotConfig, ProfileLoader, and related configuration types.
Part of the modern DI layer (app/).

Types:
    BotConfig: Main bot configuration dataclass
    BotRiskConfig: Risk/position sizing configuration
    MeanReversionConfig: Mean reversion strategy configuration
    StrategyType: Literal type for strategy selection
    ProfileLoader: YAML-first profile configuration loader
    ProfileSchema: Complete profile schema dataclass
    TradingConfig, StrategyConfig, RiskConfig, ExecutionConfig,
    SessionConfig, MonitoringConfig: Profile section dataclasses

Constants:
    DEFAULT_SPOT_RISK_PATH: Default path for spot risk config
    DEFAULT_SPOT_SYMBOLS: Default symbols for spot trading
    TOP_VOLUME_BASES: Top volume base currencies

Usage:
    from gpt_trader.app.config import BotConfig, BotRiskConfig

    config = BotConfig.from_env()
    # or
    config = BotConfig.from_profile(Profile.DEV, dry_run=True)

    # Profile loading
    from gpt_trader.app.config import ProfileLoader, load_profile
    loader = ProfileLoader()
    schema = loader.load(Profile.DEV)
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
from gpt_trader.app.config.profile_loader import (
    ExecutionConfig,
    MonitoringConfig,
    ProfileLoader,
    ProfileSchema,
    RiskConfig,
    SessionConfig,
    StrategyConfig,
    TradingConfig,
    get_profile_loader,
    load_profile,
)

__all__ = [
    # Bot configuration
    "BotConfig",
    "BotRiskConfig",
    "MeanReversionConfig",
    "StrategyType",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
    # Profile loading
    "ExecutionConfig",
    "MonitoringConfig",
    "ProfileLoader",
    "ProfileSchema",
    "RiskConfig",
    "SessionConfig",
    "StrategyConfig",
    "TradingConfig",
    "get_profile_loader",
    "load_profile",
]
