"""
Simplified Base Engine & Context.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from gpt_trader.orchestration.configuration import BotConfig

if TYPE_CHECKING:
    from gpt_trader.features.brokerages.core.protocols import (
        BrokerProtocol,
    )  # Protocol import is OK
    from gpt_trader.features.live_trade.risk.protocols import RiskManagerProtocol
    from gpt_trader.monitoring.notifications.service import NotificationService
    from gpt_trader.orchestration.protocols import (
        EventStoreProtocol,
        RuntimeStateProtocol,
        ServiceRegistryProtocol,
    )


@dataclass
class CoordinatorContext:
    config: BotConfig
    registry: ServiceRegistryProtocol | None = None
    broker: BrokerProtocol | None = None
    symbols: tuple[str, ...] = ()
    runtime_state: RuntimeStateProtocol | None = None
    risk_manager: RiskManagerProtocol | None = None
    event_store: EventStoreProtocol | None = None
    notification_service: NotificationService | None = None
    bot_id: str = ""

    def with_updates(self, **overrides: Any) -> CoordinatorContext:
        return replace(self, **overrides)


@dataclass
class HealthStatus:
    healthy: bool
    component: str
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class BaseEngine:
    def __init__(self, context: CoordinatorContext) -> None:
        self._context = context
        self._background_tasks: list[asyncio.Task[Any]] = []
        self._shutdown_complete = False

    @property
    def context(self) -> CoordinatorContext:
        return self._context

    def update_context(self, context: CoordinatorContext) -> None:
        self._context = context

    def initialize(self, context: CoordinatorContext) -> CoordinatorContext:
        return context

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        return []

    async def shutdown(self) -> None:
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
        return HealthStatus(healthy=not self._shutdown_complete, component="base")

    def _register_background_task(self, task: asyncio.Task[Any]) -> None:
        self._background_tasks.append(task)


__all__ = ["BaseEngine", "CoordinatorContext", "HealthStatus"]
