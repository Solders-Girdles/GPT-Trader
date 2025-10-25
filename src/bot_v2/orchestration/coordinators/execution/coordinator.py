"""Composite execution coordinator assembled from focused mixins."""

from __future__ import annotations

from ..base import BaseCoordinator
from .background import ExecutionCoordinatorBackgroundMixin
from .core import ExecutionCoordinatorCoreMixin
from .orders import ExecutionCoordinatorOrderMixin
from .positions import ExecutionCoordinatorPositionMixin


class ExecutionCoordinator(
    ExecutionCoordinatorOrderMixin,
    ExecutionCoordinatorPositionMixin,
    ExecutionCoordinatorBackgroundMixin,
    ExecutionCoordinatorCoreMixin,
    BaseCoordinator,
):
    """Main execution coordinator orchestrating order placement and monitoring."""


__all__ = ["ExecutionCoordinator"]
