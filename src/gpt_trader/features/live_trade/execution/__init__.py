"""Order execution components for live trading.

This module provides order validation, guard management,
and execution services for live trading.
"""

from .broker_executor import BrokerExecutor
from .decision_trace import OrderDecisionTrace
from .guard_manager import GuardManager
from .guards import RuntimeGuardState
from .order_event_recorder import OrderEventRecorder
from .order_submission import OrderSubmitter
from .state_collection import StateCollector
from .submission_result import OrderSubmissionResult, OrderSubmissionStatus
from .validation import (
    OrderValidator,
    ValidationFailureTracker,
    configure_failure_tracker,
    get_failure_tracker,
    get_validation_metrics,
)

__all__ = [
    "OrderDecisionTrace",
    # Submission
    "BrokerExecutor",
    "OrderEventRecorder",
    "OrderSubmitter",
    "OrderSubmissionResult",
    "OrderSubmissionStatus",
    "StateCollector",
    # Validation
    "OrderValidator",
    "ValidationFailureTracker",
    "configure_failure_tracker",
    "get_failure_tracker",
    "get_validation_metrics",
    # Guards
    "GuardManager",
    "RuntimeGuardState",
]
