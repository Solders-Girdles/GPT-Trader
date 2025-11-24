"""
Configuration module.
"""
from gpt_trader.orchestration.configuration.bot_config import BotConfig, config
from gpt_trader.orchestration.configuration.risk.model import (
    RiskConfig,
    RISK_CONFIG_ENV_KEYS,
    RISK_CONFIG_ENV_ALIASES,
)
from gpt_trader.orchestration.configuration.validation import (
    ConfigValidationError,
    ConfigValidationResult,
)
from gpt_trader.orchestration.configuration.core import Profile

__all__ = [
    "BotConfig",
    "config",
    "RiskConfig",
    "RISK_CONFIG_ENV_KEYS",
    "RISK_CONFIG_ENV_ALIASES",
    "ConfigValidationError",
    "ConfigValidationResult",
    "Profile",
]
