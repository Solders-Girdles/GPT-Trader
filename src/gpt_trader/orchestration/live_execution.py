"""
DEPRECATED: LiveExecutionEngine has moved to gpt_trader.features.live_trade.execution.engine

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.execution.engine import LiveExecutionEngine, LiveOrder
"""

import warnings

from gpt_trader.features.live_trade.execution.engine import LiveExecutionEngine, LiveOrder

warnings.warn(
    "gpt_trader.orchestration.live_execution is deprecated. "
    "Import from gpt_trader.features.live_trade.execution.engine instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["LiveExecutionEngine", "LiveOrder"]
