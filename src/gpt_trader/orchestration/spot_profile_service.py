"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.orchestrator.spot_profile_service

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.spot_profile_service import SpotProfileService

    # New (preferred)
    from gpt_trader.features.live_trade.orchestrator.spot_profile_service import SpotProfileService
"""

from __future__ import annotations

import warnings

# Re-export all symbols from canonical location
from gpt_trader.features.live_trade.orchestrator.spot_profile_service import (
    SpotProfileService,
    _load_spot_profile,
)

__all__ = ["SpotProfileService", "_load_spot_profile"]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.spot_profile_service is deprecated. "
    "Import from gpt_trader.features.live_trade.orchestrator.spot_profile_service instead.",
    DeprecationWarning,
    stacklevel=2,
)
