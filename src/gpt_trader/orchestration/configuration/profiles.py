"""
DEPRECATED: This module has moved to gpt_trader.app.config.profile_loader

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.configuration.profiles import build_profile_config

    # New (preferred)
    from gpt_trader.app.config.profile_loader import build_profile_config
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.app.config.profile_loader import (
    ProfileLoader,
    ProfileSchema,
    build_profile_config,
    load_profile,
)

__all__ = [
    "ProfileLoader",
    "ProfileSchema",
    "build_profile_config",
    "load_profile",
]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.configuration.profiles is deprecated. "
    "Import from gpt_trader.app.config.profile_loader instead.",
    DeprecationWarning,
    stacklevel=2,
)
