"""Modular BotConfig components."""

from .bot_config import BotConfig
from .defaults import DEFAULT_SPOT_RISK_PATH, DEFAULT_SPOT_SYMBOLS, TOP_VOLUME_BASES
from .state import ConfigState

__all__ = [
    "BotConfig",
    "ConfigState",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
]
