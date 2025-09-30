"""
Execution engine submodules for live trading.

This package breaks down the LiveExecutionEngine into focused components:
- guards: Runtime safety guards
- validation: Pre-trade validation logic
- order_submission: Order placement and recording
- state_collection: Account state collection helpers
"""

from bot_v2.orchestration.execution.guards import GuardManager, RuntimeGuardState
from bot_v2.orchestration.execution.order_submission import OrderSubmitter
from bot_v2.orchestration.execution.state_collection import StateCollector
from bot_v2.orchestration.execution.validation import OrderValidator

__all__ = [
    "GuardManager",
    "OrderValidator",
    "OrderSubmitter",
    "StateCollector",
    "RuntimeGuardState",
]
