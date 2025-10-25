"""Configuration package for GPT-Trader orchestration.

Provides the primary configuration model, manager, and validation helpers while
preserving the original import surface from the monolithic module.
"""

# Direct import to avoid legacy config module dependencies
try:
    from bot_v2.config.types import Profile
except ImportError:
    # Fallback definition if legacy config module has issues
    from enum import Enum

    class Profile(str, Enum):
        LIVE = "live"
        PAPER = "paper"
        CANARY = "canary"
        SPOT = "spot"


from .core import DEFAULT_SPOT_RISK_PATH, DEFAULT_SPOT_SYMBOLS, TOP_VOLUME_BASES, BotConfig
from .manager import ConfigManager
from .risk import (
    DEFAULT_RISK_CONFIG_PATH,
    RISK_CONFIG_ENV_ALIASES,
    RISK_CONFIG_ENV_KEYS,
    RiskConfig,
)
from .validation import ConfigValidationError, ConfigValidationResult

__all__ = [
    "BotConfig",
    "ConfigManager",
    "ConfigValidationError",
    "ConfigValidationResult",
    "DEFAULT_RISK_CONFIG_PATH",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "Profile",
    "RISK_CONFIG_ENV_ALIASES",
    "RISK_CONFIG_ENV_KEYS",
    "RiskConfig",
    "TOP_VOLUME_BASES",
]
