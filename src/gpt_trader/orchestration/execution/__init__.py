"""
DEPRECATED: This module has moved to gpt_trader.features.live_trade.execution

This shim re-exports all symbols for backwards compatibility.
Please update your imports to use the new location:

    # Old (deprecated)
    from gpt_trader.orchestration.execution import GuardManager

    # New (preferred)
    from gpt_trader.features.live_trade.execution import GuardManager
"""

from __future__ import annotations

import warnings

from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor
from gpt_trader.features.live_trade.execution.guard_manager import GuardManager
from gpt_trader.features.live_trade.execution.guards import RuntimeGuardState
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.features.live_trade.execution.state_collection import StateCollector
from gpt_trader.features.live_trade.execution.validation import (
    OrderValidator,
    ValidationFailureTracker,
    get_validation_metrics,
)

__all__ = [
    "BrokerExecutor",
    "GuardManager",
    "OrderEventRecorder",
    "OrderSubmitter",
    "OrderValidator",
    "RuntimeGuardState",
    "StateCollector",
    "ValidationFailureTracker",
    "get_validation_metrics",
]

# Emit deprecation warning on import
warnings.warn(
    "gpt_trader.orchestration.execution is deprecated. "
    "Import from gpt_trader.features.live_trade.execution instead.",
    DeprecationWarning,
    stacklevel=2,
)
