"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.mark_staleness
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.mark_staleness import MarkStalenessGuard

__all__ = ["MarkStalenessGuard"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.mark_staleness is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.mark_staleness instead.",
    DeprecationWarning,
    stacklevel=2,
)
