"""
Execution engine submodules for live trading.

This package breaks down the LiveExecutionEngine into focused components:
- guards: Runtime safety guards
- validation: Pre-trade validation logic
- order_submission: Order placement and recording
- state_collection: Account state collection helpers
- engine_factory: Factory for creating execution engines
- order_placement: Service for translating decisions to orders
- runtime_supervisor: Supervisor for background execution tasks
"""

from bot_v2.orchestration.execution.engine_factory import ExecutionEngineFactory
from bot_v2.orchestration.execution.guards import GuardManager, RuntimeGuardState
from bot_v2.orchestration.execution.order_placement import OrderPlacementService
from bot_v2.orchestration.execution.order_submission import OrderSubmitter
from bot_v2.orchestration.execution.runtime_supervisor import ExecutionRuntimeSupervisor
from bot_v2.orchestration.execution.state_collection import StateCollector
from bot_v2.orchestration.execution.validation import OrderValidator

__all__ = [
    "ExecutionEngineFactory",
    "ExecutionRuntimeSupervisor",
    "GuardManager",
    "OrderPlacementService",
    "OrderSubmitter",
    "OrderValidator",
    "RuntimeGuardState",
    "StateCollector",
]
