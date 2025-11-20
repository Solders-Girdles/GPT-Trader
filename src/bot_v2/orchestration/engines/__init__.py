"""Coordinator abstractions for the PerpsBot orchestration layer.

The coordinator pattern provides a consistent lifecycle contract around three primitives:

- :class:`CoordinatorContext` – immutable dependency snapshot shared between coordinators
- :class:`BaseEngine` – convenience base with default implementations and task tracking
- :class:`CoordinatorRegistry` – registration and lifecycle orchestrator (init/start/shutdown/health)

Typical usage::

    context = CoordinatorContext(config=bot.config, registry=bot.registry, ...)
    registry = CoordinatorRegistry(context)
    registry.register(RuntimeEngine(context))
    ...
    updated_context = registry.initialize_all()
    await registry.start_all_background_tasks()

Legacy facade modules under ``bot_v2/orchestration/*_coordinator.py`` continue to wrap these
implementations for backwards compatibility; new features should import engines from this
package directly. See ADR 002 for detailed rationale and migration notes.
"""

from .base import BaseEngine, Coordinator, CoordinatorContext, HealthStatus
from .execution import ExecutionEngine
from .registry import CoordinatorRegistry
from .runtime import BrokerBootstrapError, RuntimeEngine
from .strategy import TradingEngine
from .telemetry_coordinator import TelemetryEngine

__all__ = [
    "BaseEngine",
    "BrokerBootstrapError",
    "Coordinator",
    "CoordinatorContext",
    "CoordinatorRegistry",
    "ExecutionEngine",
    "HealthStatus",
    "RuntimeEngine",
    "TradingEngine",
    "TelemetryEngine",
]
