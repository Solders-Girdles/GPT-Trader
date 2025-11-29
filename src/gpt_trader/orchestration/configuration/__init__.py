"""
Configuration module.

Provides:
- BotConfig: Main bot configuration dataclass
- Profile: Configuration profile enum (dev, demo, prod, canary, spot, test)
- ProfileSchema: Structured profile configuration from YAML
- ProfileLoader: YAML-first profile loading with fallbacks
"""

from gpt_trader.config.types import Profile
from gpt_trader.orchestration.configuration.bot_config import (
    TOP_VOLUME_BASES,
    BotConfig,
    BotRiskConfig,
)
from gpt_trader.orchestration.configuration.profile_loader import (
    ProfileLoader,
    ProfileSchema,
    load_profile,
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
    "BotRiskConfig",
    "ConfigValidationError",
    "ConfigValidationResult",
    "Profile",
    "ProfileLoader",
    "ProfileSchema",
    "RISK_CONFIG_ENV_ALIASES",
    "RISK_CONFIG_ENV_KEYS",
    "RiskConfig",
    "TOP_VOLUME_BASES",
    "load_profile",
]
