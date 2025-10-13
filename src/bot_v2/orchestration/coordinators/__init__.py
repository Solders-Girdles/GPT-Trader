"""Coordinator abstractions for the PerpsBot orchestration layer.

The coordinator pattern provides a consistent lifecycle contract around three primitives:

- :class:`CoordinatorContext` – immutable dependency snapshot shared between coordinators
- :class:`BaseCoordinator` – convenience base with default implementations and task tracking
- :class:`CoordinatorRegistry` – registration and lifecycle orchestrator (init/start/shutdown/health)

Typical usage::

    context = CoordinatorContext(config=bot.config, registry=bot.registry, ...)
    registry = CoordinatorRegistry(context)
    registry.register(RuntimeCoordinator(context))
    ...
    updated_context = registry.initialize_all()
    await registry.start_all_background_tasks()

Legacy facade modules under ``bot_v2/orchestration/*_coordinator.py`` continue to wrap these
implementations for backwards compatibility; new features should import coordinators from this
package directly. See ADR 002 for detailed rationale and migration notes.
"""

from .base import BaseCoordinator, Coordinator, CoordinatorContext, HealthStatus
from .execution import ExecutionCoordinator
from .registry import CoordinatorRegistry
from .runtime import RuntimeCoordinator
from .strategy import StrategyCoordinator
from .telemetry import TelemetryCoordinator

__all__ = [
    "BaseCoordinator",
    "Coordinator",
    "CoordinatorContext",
    "CoordinatorRegistry",
    "ExecutionCoordinator",
    "HealthStatus",
    "RuntimeCoordinator",
    "StrategyCoordinator",
    "TelemetryCoordinator",
]
