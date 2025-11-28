"""Modular BotConfig components."""

from .bot_config import (
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
    BotConfig,
    BotRiskConfig,
    ConfigState,
    config,
)

__all__ = [
    "BotConfig",
    "BotRiskConfig",
    "config",
    "ConfigState",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
]
