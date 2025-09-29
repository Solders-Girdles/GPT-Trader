"""Bootstrap helpers for wiring GPT-Trader orchestration components."""

from __future__ import annotations

from typing import Any

from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry


def build_service_registry(
    config: BotConfig, registry: ServiceRegistry | None = None
) -> ServiceRegistry:
    """Return a service registry seeded for the supplied configuration."""

    if registry is not None:
        if registry.config is config:
            return registry
        return registry.with_updates(config=config)
    return empty_registry(config)


def build_bot(
    config: BotConfig, registry: ServiceRegistry | None = None
) -> tuple[PerpsBot, ServiceRegistry]:
    """Instantiate a PerpsBot with a populated service registry."""

    service_registry = build_service_registry(config, registry)
    bot = PerpsBot(config, registry=service_registry)
    # PerpsBot may have enriched the registry with lazily constructed services
    return bot, bot.registry


def bot_from_profile(profile: str, **overrides: Any) -> tuple[PerpsBot, ServiceRegistry]:
    """Convenience helper for CLI/bootstrap callers."""

    config = BotConfig.from_profile(profile, **overrides)
    return build_bot(config)
