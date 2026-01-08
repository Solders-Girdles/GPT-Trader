"""
Backward compatibility re-export for DegradationState.

The canonical location is now:
    gpt_trader.features.live_trade.degradation

This module re-exports DegradationState and PauseRecord for backward
compatibility with existing imports from orchestration.execution.degradation.
"""

# Re-export from canonical location
from gpt_trader.features.live_trade.degradation import DegradationState, PauseRecord

__all__ = ["DegradationState", "PauseRecord"]
