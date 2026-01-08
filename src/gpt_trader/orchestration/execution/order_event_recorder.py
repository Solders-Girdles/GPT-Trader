"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution.order_event_recorder
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder

__all__ = ["OrderEventRecorder"]

warnings.warn(
    "gpt_trader.orchestration.execution.order_event_recorder is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.order_event_recorder instead.",
    DeprecationWarning,
    stacklevel=2,
)
