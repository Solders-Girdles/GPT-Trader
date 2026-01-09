"""
DEPRECATED: This module has moved to gpt_trader.app.config.defaults

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.configuration.bot_config.defaults import TOP_VOLUME_BASES

    # New (preferred)
    from gpt_trader.app.config.defaults import TOP_VOLUME_BASES
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.app.config.defaults import (
    DEFAULT_SPOT_RISK_PATH,
    DEFAULT_SPOT_SYMBOLS,
    TOP_VOLUME_BASES,
)

__all__ = [
    "TOP_VOLUME_BASES",
    "DEFAULT_SPOT_SYMBOLS",
    "DEFAULT_SPOT_RISK_PATH",
]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.configuration.bot_config.defaults is deprecated. "
    "Import from gpt_trader.app.config.defaults instead.",
    DeprecationWarning,
    stacklevel=2,
)
