"""Configuration package for GPT-Trader orchestration.

Provides the primary configuration model, manager, and validation helpers while
preserving the original import surface from the monolithic module.
"""

from bot_v2.config.types import Profile

from .core import DEFAULT_SPOT_RISK_PATH, DEFAULT_SPOT_SYMBOLS, TOP_VOLUME_BASES, BotConfig
from .manager import ConfigManager
from .validation import ConfigValidationError, ConfigValidationResult

__all__ = [
    "BotConfig",
    "ConfigManager",
    "ConfigValidationError",
    "ConfigValidationResult",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "Profile",
    "TOP_VOLUME_BASES",
]
