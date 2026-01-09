"""
DEPRECATED: This module has moved to gpt_trader.app.bootstrap

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.bootstrap import build_bot, bot_from_profile

    # New (preferred)
    from gpt_trader.app.bootstrap import build_bot, bot_from_profile
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.app.bootstrap import (
    BootstrapLogRecord,
    RuntimePaths,
    bot_from_profile,
    build_bot,
    normalise_symbols,
    resolve_runtime_paths,
)

__all__ = [
    "BootstrapLogRecord",
    "normalise_symbols",
    "resolve_runtime_paths",
    "RuntimePaths",
    "build_bot",
    "bot_from_profile",
]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.bootstrap is deprecated. "
    "Import from gpt_trader.app.bootstrap instead.",
    DeprecationWarning,
    stacklevel=2,
)
