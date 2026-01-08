"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.daily_loss
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.daily_loss import DailyLossGuard

__all__ = ["DailyLossGuard"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.daily_loss is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.daily_loss instead.",
    DeprecationWarning,
    stacklevel=2,
)
