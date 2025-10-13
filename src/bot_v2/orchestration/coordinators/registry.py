"""Coordinator registry for orchestrating lifecycle management."""

from __future__ import annotations

import logging
from typing import Any

from .base import Coordinator, CoordinatorContext, HealthStatus

logger = logging.getLogger(__name__)


class CoordinatorRegistry:
    """Manages coordinator registration, lifecycle, and health checks."""

    def __init__(self, context: CoordinatorContext) -> None:
        self._context = context
        self._coordinators: dict[str, Coordinator] = {}
        self._initialization_order: list[str] = []

    @property
    def context(self) -> CoordinatorContext:
        """Return the latest coordinator context."""

        return self._context

    def register(self, coordinator: Coordinator) -> None:
        """Register a coordinator for lifecycle management."""

        name = coordinator.name
        if name in self._coordinators:
            raise ValueError(f"Coordinator '{name}' already registered")

        self._coordinators[name] = coordinator
        self._initialization_order.append(name)
        logger.debug("Registered coordinator: %s", name)

    def get(self, name: str) -> Coordinator | None:
        """Retrieve a coordinator by name."""

        return self._coordinators.get(name)

    def initialize_all(self) -> CoordinatorContext:
        """Run initialization for all registered coordinators."""

        context = self._context
        for name in self._initialization_order:
            coordinator = self._coordinators[name]
            logger.info("Initializing coordinator: %s", name)
            context = coordinator.initialize(context)
            self._propagate_context(context)

        self._context = context
        return context

    async def start_all_background_tasks(self) -> list[Any]:
        """Start background tasks for all coordinators."""

        tasks: list[Any] = []
        for name, coordinator in self._coordinators.items():
            logger.info("Starting background tasks for coordinator: %s", name)
            started = await coordinator.start_background_tasks()
            tasks.extend(started)
        return tasks

    async def shutdown_all(self) -> None:
        """Shutdown coordinators in reverse initialization order."""

        for name in reversed(self._initialization_order):
            coordinator = self._coordinators[name]
            logger.info("Shutting down coordinator: %s", name)
            await coordinator.shutdown()

    def health_check_all(self) -> dict[str, HealthStatus]:
        """Collect health status from all coordinators."""

        return {
            name: coordinator.health_check() for name, coordinator in self._coordinators.items()
        }

    def _propagate_context(self, context: CoordinatorContext) -> None:
        """Notify all coordinators about context updates."""

        for coordinator in self._coordinators.values():
            update_fn = getattr(coordinator, "update_context", None)
            if callable(update_fn):
                update_fn(context)


__all__ = ["CoordinatorRegistry"]
