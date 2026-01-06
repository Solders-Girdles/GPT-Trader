"""
Execution engine submodules for live trading.

This package breaks down the LiveExecutionEngine into focused components:
- guards: Runtime safety guards
- validation: Pre-trade validation logic
- order_submission: Order placement and recording
- state_collection: Account state collection helpers

Internal modules (used by order_submission):
- order_event_recorder: Event recording and telemetry
- broker_executor: Broker communication and integration mode handling
"""

from gpt_trader.orchestration.execution.guard_manager import GuardManager
from gpt_trader.orchestration.execution.guards import RuntimeGuardState
from gpt_trader.orchestration.execution.order_submission import OrderSubmitter
from gpt_trader.orchestration.execution.state_collection import StateCollector
from gpt_trader.orchestration.execution.validation import (
    OrderValidator,
    ValidationFailureTracker,
    get_validation_metrics,
)

__all__ = [
    "GuardManager",
    "OrderValidator",
    "OrderSubmitter",
    "StateCollector",
    "RuntimeGuardState",
    "ValidationFailureTracker",
    "get_validation_metrics",
]
