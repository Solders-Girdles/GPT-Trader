"""Helpers for constructing coordinator contexts from bot state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.bot_state_extraction import BotStateExtractor
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.service_registry import ServiceRegistry

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from bot_v2.orchestration.perps_bot import PerpsBot

__all__ = [
    "ensure_service_registry",
    "build_coordinator_context",
]


def ensure_service_registry(
    bot: PerpsBot,
    registry: ServiceRegistry | None = None,
) -> ServiceRegistry:
    """Return a ServiceRegistry instance aligned with the bot configuration."""

    return BotStateExtractor(bot).service_registry(registry)


def build_coordinator_context(
    bot: PerpsBot,
    *,
    registry: ServiceRegistry | None = None,
    overrides: dict[str, Any] | None = None,
) -> CoordinatorContext:
    """Build a CoordinatorContext snapshot from a PerpsBot instance."""

    extractor = BotStateExtractor(bot)
    resolved_registry = extractor.service_registry(registry)
    context_data = extractor.coordinator_context_kwargs(resolved_registry, overrides=overrides)

    return CoordinatorContext(**context_data)
