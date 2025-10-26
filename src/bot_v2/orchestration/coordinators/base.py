"""Base abstractions for PerpsBot coordinators.

These primitives define the coordinator pattern introduced in 2025-10. Coordinators consume an
immutable :class:`CoordinatorContext`, expose a uniform lifecycle contract, and delegate lifecycle
management to :class:`bot_v2.orchestration.coordinators.registry.CoordinatorRegistry`.

Most concrete coordinators should:

1. Inherit from :class:`BaseCoordinator`.
2. Override :py:meth:`initialize` to build dependencies and return an updated context.
3. Optionally implement :py:meth:`start_background_tasks` to launch long-running jobs (register tasks
   with :py:meth:`_register_background_task` for automatic shutdown).
4. Provide a :py:meth:`health_check` with domain-specific insights beyond the default status.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from asyncio import Task

    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.features.live_trade.risk import LiveRiskManager
    from bot_v2.orchestration.configuration import BotConfig
    from bot_v2.orchestration.service_registry import ServiceRegistry
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore


@dataclass
class CoordinatorContext:
    """Shared dependency snapshot exchanged between coordinators.

    The context exposes only the dependencies coordinators actively use, reducing coupling to the
    full `PerpsBot` implementation. Instances are immutable; overridden copies must be produced via
    :py:meth:`with_updates` so the registry can broadcast new snapshots safely.
    """

    config: BotConfig
    registry: ServiceRegistry
    event_store: EventStore | None = None
    orders_store: OrdersStore | None = None
    broker: IBrokerage | None = None
    risk_manager: LiveRiskManager | None = None
    symbols: tuple[str, ...] = ()
    bot_id: str = "coinbase_trader"
    runtime_state: Any = None
    config_controller: Any | None = None
    strategy_orchestrator: Any | None = None
    strategy_coordinator: Any | None = None
    execution_coordinator: Any | None = None
    product_cache: dict[str, Any] | None = None
    session_guard: Any | None = None
    configuration_guardian: Any | None = None
    system_monitor: Any | None = None
    set_reduce_only_mode: Callable[[bool, str], None] | None = None
    shutdown_hook: Callable[[], Awaitable[None]] | None = None
    set_running_flag: Callable[[bool], None] | None = None

    def with_updates(self, **overrides: Any) -> CoordinatorContext:
        """Return a new context with selected fields overridden."""

        return replace(self, **overrides)


@dataclass(slots=True)
class HealthStatus:
    """Coordinator health check result."""

    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Coordinator(Protocol):
    """Lifecycle contract surfaced by all orchestration coordinators."""

    @property
    def name(self) -> str:
        """Coordinator name used for logging and registry lookups."""

    def initialize(self, context: CoordinatorContext) -> CoordinatorContext:
        """Perform synchronous initialization and return updated context."""

    def update_context(self, context: CoordinatorContext) -> None:
        """Receive context updates produced by other coordinators."""

    async def start_background_tasks(self) -> list[Task[Any]]:
        """Start any asynchronous background work and return tracking tasks."""

    async def shutdown(self) -> None:
        """Release resources and stop background activities."""

    def health_check(self) -> HealthStatus:
        """Return the current health status for monitoring."""


class BaseCoordinator(ABC):
    """Abstract base class implementing the coordinator lifecycle contract."""

    def __init__(self, context: CoordinatorContext) -> None:
        self._context = context
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._shutdown_complete = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Coordinator identifier used for logging."""

    @property
    def context(self) -> CoordinatorContext:
        """Return the latest coordinator context."""

        return self._context

    def update_context(self, context: CoordinatorContext) -> None:
        """Update internal context reference (called by registry)."""

        self._context = context

    def initialize(self, context: CoordinatorContext) -> CoordinatorContext:
        """Perform synchronous initialisation and return an updated context snapshot."""

        return context

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Default implementation does not spawn background work."""

        return []

    async def shutdown(self) -> None:
        """Cancel any registered background tasks."""

        if self._shutdown_complete:
            return

        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_tasks.clear()
        self._shutdown_complete = True

    def health_check(self) -> HealthStatus:
        """Default health status derived from shutdown state."""

        return HealthStatus(
            healthy=not self._shutdown_complete,
            component=self.name,
            details={
                "background_tasks": len(self._background_tasks),
                "shutdown_complete": self._shutdown_complete,
            },
        )

    def _register_background_task(self, task: asyncio.Task[Any]) -> None:
        """Track a background task for coordinated shutdown."""

        self._background_tasks.append(task)


__all__ = [
    "BaseCoordinator",
    "Coordinator",
    "CoordinatorContext",
    "HealthStatus",
]
