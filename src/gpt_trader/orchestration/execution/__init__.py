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

NOTE: This module re-exports from features/live_trade/execution/ for
backward compatibility. Import from gpt_trader.features.live_trade.execution
for the canonical location.
"""

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
