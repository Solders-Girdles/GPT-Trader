"""Bootstrap helpers for wiring GPT-Trader orchestration components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bot_builder import create_perps_bot
from bot_v2.orchestration.runtime_settings import RuntimeSettingsProvider
from bot_v2.orchestration.service_registry import ServiceRegistry, empty_registry

if TYPE_CHECKING:  # pragma: no cover - circular type import guard
    from bot_v2.orchestration.perps_bot import PerpsBot


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
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
) -> tuple[PerpsBot, ServiceRegistry]:
    """Instantiate a PerpsBot with a populated service registry."""

    service_registry = build_service_registry(config, registry)
    bot = create_perps_bot(
        config,
        service_registry,
        settings_provider=settings_provider,
    )
    return bot, bot.registry


def bot_from_profile(
    profile: str,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
    **overrides: Any,
) -> tuple[PerpsBot, ServiceRegistry]:
    """Convenience helper for CLI/bootstrap callers."""

    config = BotConfig.from_profile(profile, **overrides)
    return build_bot(config, settings_provider=settings_provider)
