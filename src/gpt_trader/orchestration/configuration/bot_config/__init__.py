"""
DEPRECATED: This module has moved to gpt_trader.app.config

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.configuration.bot_config import BotConfig

    # New (preferred)
    from gpt_trader.app.config import BotConfig
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.app.config import (
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
    BotConfig,
    BotRiskConfig,
)

__all__ = [
    "BotConfig",
    "BotRiskConfig",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.configuration.bot_config is deprecated. "
    "Import from gpt_trader.app.config instead.",
    DeprecationWarning,
    stacklevel=2,
)
