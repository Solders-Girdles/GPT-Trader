"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.guards.protocol
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.guards.protocol import (
    Guard,
    RuntimeGuardState,
)

__all__ = ["Guard", "RuntimeGuardState"]

warnings.warn(
    "gpt_trader.orchestration.execution.guards.protocol is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.guards.protocol instead.",
    DeprecationWarning,
    stacklevel=2,
)
