"""
Configuration module.
"""

from gpt_trader.config.types import Profile
from gpt_trader.orchestration.configuration.bot_config import (
    TOP_VOLUME_BASES,
    BotConfig,
    config,
)
from gpt_trader.orchestration.configuration.risk.model import (
    RISK_CONFIG_ENV_ALIASES,
    RISK_CONFIG_ENV_KEYS,
    RiskConfig,
)
from gpt_trader.orchestration.configuration.validation import (
    ConfigValidationError,
    ConfigValidationResult,
)

__all__ = [
    "BotConfig",
    "config",
    "RiskConfig",
    "RISK_CONFIG_ENV_KEYS",
    "RISK_CONFIG_ENV_ALIASES",
    "ConfigValidationError",
    "ConfigValidationResult",
    "Profile",
    "TOP_VOLUME_BASES",
]
