"""
DEPRECATED: This module has moved to gpt_trader.app.config.profile_loader

This shim exists for backwards compatibility. Update imports to use:
    from gpt_trader.app.config.profile_loader import ProfileLoader, ProfileSchema
    # or
    from gpt_trader.app.config import ProfileLoader, ProfileSchema, load_profile
"""

from __future__ import annotations

import warnings

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

warnings.warn(
    "gpt_trader.orchestration.configuration.profile_loader is deprecated. "
    "Import from gpt_trader.app.config.profile_loader instead.",
    DeprecationWarning,
    stacklevel=2,
)
