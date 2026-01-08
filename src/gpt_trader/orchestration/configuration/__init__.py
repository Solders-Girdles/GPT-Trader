"""
Configuration module.

DEPRECATION NOTICE: BotConfig and related types have moved to gpt_trader.app.config.
This module re-exports them for backwards compatibility.

Update imports to use:
    from gpt_trader.app.config import BotConfig, BotRiskConfig

Provides:
- BotConfig: Main bot configuration dataclass
- Profile: Configuration profile enum (dev, demo, prod, canary, spot, test)
- ProfileSchema: Structured profile configuration from YAML
- ProfileLoader: YAML-first profile loading with fallbacks
"""

from gpt_trader.app.config import (
    TOP_VOLUME_BASES,
    BotConfig,
    BotRiskConfig,
)
from gpt_trader.app.config.profile_loader import (
    ProfileLoader,
    ProfileSchema,
    load_profile,
)
from gpt_trader.config.types import Profile
from gpt_trader.features.live_trade.risk.config import (
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
