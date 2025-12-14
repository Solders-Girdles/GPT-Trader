"""
TUI State Management Package.

This package provides state validation, delta update, and trade matching
capabilities for the TUI.

Modules:
    validators: State validation layer for incoming bot status updates
    delta_updater: Efficient delta-based state updates to minimize UI flicker
    trade_matcher_state: State dataclass for trade matching operations
"""

from gpt_trader.tui.state_management.delta_updater import StateDeltaUpdater
from gpt_trader.tui.state_management.trade_matcher_state import TradeMatcherState
from gpt_trader.tui.state_management.validators import StateValidator, ValidationResult

__all__ = [
    "StateDeltaUpdater",
    "StateValidator",
    "TradeMatcherState",
    "ValidationResult",
]
