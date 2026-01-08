"""Order execution components for live trading.

This module provides order routing, validation, guard management,
and execution services for multi-venue trading.
"""

from .broker_executor import BrokerExecutor
from .guard_manager import GuardManager
from .guards import RuntimeGuardState
from .order_event_recorder import OrderEventRecorder
from .order_submission import OrderSubmitter
from .router import OrderResult, OrderRouter
from .state_collection import StateCollector
from .validation import (
    OrderValidator,
    ValidationFailureTracker,
    configure_failure_tracker,
    get_failure_tracker,
    get_validation_metrics,
)

__all__ = [
    # Routing
    "OrderRouter",
    "OrderResult",
    # Submission
    "BrokerExecutor",
    "OrderEventRecorder",
    "OrderSubmitter",
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
