"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guard_manager

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.execution.guard_manager import GuardManager

    # New (preferred)
    from gpt_trader.features.live_trade.execution.guard_manager import GuardManager
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guard_manager import GuardManager

__all__ = ["GuardManager"]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.execution.guard_manager is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guard_manager instead.",
    DeprecationWarning,
    stacklevel=2,
)
