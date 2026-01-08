"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.cache
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.cache import GuardStateCache

__all__ = ["GuardStateCache"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.cache is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.cache instead.",
    DeprecationWarning,
    stacklevel=2,
)
