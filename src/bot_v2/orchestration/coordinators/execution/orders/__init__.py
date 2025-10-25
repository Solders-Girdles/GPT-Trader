"""Order management mixins for execution coordinator."""

from .execution import DecisionExecutionMixin
from .lifecycle import OrderLifecycleMixin
from .lock import OrderLockMixin
from .placement import OrderPlacementMixin


class ExecutionCoordinatorOrderMixin(
    OrderPlacementMixin,
    OrderLifecycleMixin,
    OrderLockMixin,
    DecisionExecutionMixin,
):
    """Combined order mixin for the execution coordinator."""


__all__ = [
    "ExecutionCoordinatorOrderMixin",
    "OrderPlacementMixin",
    "OrderLifecycleMixin",
    "OrderLockMixin",
    "DecisionExecutionMixin",
]
