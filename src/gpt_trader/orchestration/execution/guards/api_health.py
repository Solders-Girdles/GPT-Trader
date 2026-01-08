"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.api_health
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.api_health import ApiHealthGuard

__all__ = ["ApiHealthGuard"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.api_health is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.api_health instead.",
    DeprecationWarning,
    stacklevel=2,
)
