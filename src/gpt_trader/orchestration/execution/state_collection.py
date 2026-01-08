"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.state_collection
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.state_collection import StateCollector

__all__ = ["StateCollector"]

warnings.warn(
    "gpt_trader.orchestration.execution.state_collection is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.state_collection instead.",
    DeprecationWarning,
    stacklevel=2,
)
