"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.volatility
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.volatility import VolatilityGuard

__all__ = ["VolatilityGuard"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.volatility is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.volatility instead.",
    DeprecationWarning,
    stacklevel=2,
)
