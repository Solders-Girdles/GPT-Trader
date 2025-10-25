"""Bootstrap helpers for wiring GPT-Trader orchestration components.

.. deprecated::
    This module contains legacy bootstrap functions that create ServiceRegistry instances.
    Use ApplicationContainer-based approaches instead:

    Modern replacements:
    - build_bot() -> Use ApplicationContainer.create_perps_bot()
    - bot_from_profile() -> ApplicationContainer(config).create_perps_bot()

Migration guide:
    - Replace: build_bot(config, registry)
    - With: ApplicationContainer(config).create_perps_bot()
    - Replace: bot_from_profile(profile, **overrides)
    - With: ApplicationContainer(BotConfig.from_profile(profile, **overrides)).create_perps_bot()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking guard
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.orchestration.runtime_settings import RuntimeSettingsProvider
    from bot_v2.orchestration.service_registry import ServiceRegistry

# Import modern components for the new functions
from bot_v2.app.container import create_application_container
from bot_v2.orchestration.configuration import BotConfig


def build_service_registry(
    config: BotConfig, registry: ServiceRegistry | None = None
) -> ServiceRegistry:
    """
    Return a service registry seeded for the supplied configuration.

    .. deprecated::
        Use ApplicationContainer.create_service_registry() instead.
        ServiceRegistry is deprecated in favor of ApplicationContainer.
    """
    import warnings

    warnings.warn(
        "build_service_registry() is deprecated. Use ApplicationContainer instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if registry is not None:
        if hasattr(registry, "config") and registry.config is config:
            return registry
        return registry.with_updates(config=config)

    # Create minimal registry for backward compatibility
    from bot_v2.orchestration.service_registry import empty_registry

    return empty_registry(config)


def build_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
) -> tuple[PerpsBot, ServiceRegistry]:
    """
    Instantiate a PerpsBot with a populated service registry.

    .. deprecated::
        Use ApplicationContainer.create_perps_bot() instead.
        This function creates ServiceRegistry directly which is deprecated.
    """
    import warnings

    warnings.warn(
        "build_bot() is deprecated. Use ApplicationContainer(config).create_perps_bot() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # For backward compatibility, still create registry but also provide modern alternative
    from bot_v2.orchestration.perps_bot_builder import create_perps_bot

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
    """
    Convenience helper for CLI/bootstrap callers.

    .. deprecated::
        Use ApplicationContainer(BotConfig.from_profile(...)).create_perps_bot() instead.
        This function creates ServiceRegistry directly which is deprecated.
    """
    import warnings

    warnings.warn(
        "bot_from_profile() is deprecated. Use ApplicationContainer instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    config = BotConfig.from_profile(profile, **overrides)
    return build_bot(config, settings_provider=settings_provider)


# Modern replacement functions
def build_bot_modern(
    config: BotConfig,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
    **overrides: Any,
) -> PerpsBot:
    """
    Create a PerpsBot using the modern ApplicationContainer approach.

    This is the recommended replacement for build_bot().
    """
    # Get runtime settings if provider provided
    settings = None
    if settings_provider is not None:
        settings = settings_provider.get_settings()

    container = create_application_container(config, settings)
    return container.create_perps_bot(**overrides)


def bot_from_profile_modern(
    profile: str,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
    **overrides: Any,
) -> PerpsBot:
    """
    Modern replacement for bot_from_profile() using ApplicationContainer.

    This is the recommended way to create PerpsBot instances.
    """
    config = BotConfig.from_profile(profile, **overrides)
    return build_bot_modern(config, settings_provider=settings_provider)
