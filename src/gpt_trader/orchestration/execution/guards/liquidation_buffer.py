"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.liquidation_buffer
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.liquidation_buffer import (
    LiquidationBufferGuard,
)

__all__ = ["LiquidationBufferGuard"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.liquidation_buffer is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.liquidation_buffer instead.",
    DeprecationWarning,
    stacklevel=2,
)
