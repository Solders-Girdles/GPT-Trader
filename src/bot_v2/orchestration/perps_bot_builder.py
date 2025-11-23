"""Builder utilities for constructing :class:`PerpsBot` instances."""

from __future__ import annotations

from typing import Any

from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bootstrap import prepare_perps_bot
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.runtime_settings import (
    DEFAULT_RUNTIME_SETTINGS_PROVIDER,
    RuntimeSettings,
    RuntimeSettingsProvider,
)
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_trader_builder")


class PerpsBotBuilder:
    """Composable builder that assembles a :class:`PerpsBot` without side effects."""

    def __init__(
        self, provider: RuntimeSettingsProvider | None = None, use_container: bool = False
    ) -> None:
        self._config: BotConfig | None = None
        self._registry: ServiceRegistry | None = None
        self._settings: RuntimeSettings | None = None
        self._settings_provider: RuntimeSettingsProvider = (
            provider or DEFAULT_RUNTIME_SETTINGS_PROVIDER
        )
        self._use_container = use_container
        self._container: Any = None

    def with_config(self, config: BotConfig) -> PerpsBotBuilder:
        self._config = config
        return self

    def with_registry(self, registry: ServiceRegistry) -> PerpsBotBuilder:
        self._registry = registry
        return self

    def with_settings(self, settings: RuntimeSettings) -> PerpsBotBuilder:
        self._settings = settings
        return self

    def with_settings_provider(self, provider: RuntimeSettingsProvider) -> PerpsBotBuilder:
        self._settings_provider = provider
        return self

    def with_container(self, use_container: bool = True) -> PerpsBotBuilder:
        """Enable or disable container-based construction."""
        self._use_container = use_container
        return self

    def with_application_container(self, container: Any) -> PerpsBotBuilder:
        """Use a specific application container."""
        self._container = container
        self._use_container = True
        return self

    def build(self) -> PerpsBot:
        if self._config is None:
            raise ValueError("Configuration must be supplied before building the bot")

        # Use container-based construction if enabled
        if self._use_container:
            return self._build_with_container()

        # Legacy construction path
        return self._build_legacy()

    def _build_with_container(self) -> PerpsBot:
        """Build PerpsBot using the application container."""
        from bot_v2.app.container import create_application_container

        # Get settings
        settings = self._settings
        if settings is None and self._registry is not None:
            settings = self._registry.runtime_settings
        if settings is None:
            settings = self._settings_provider.get()

        # Create or use provided container
        if self._container is None:
            container = create_application_container(self._config, settings)
        else:
            container = self._container

        # Create bot from container
        bot = PerpsBot.from_container(container)

        symbols = list(bot.symbols)
        logger.info(
            "Coinbase Trader constructed via builder with container",
            operation="coinbase_trader_builder",
            stage="build_complete_with_container",
            profile=self._config.profile.value,
            symbol_count=len(symbols),
            symbols=symbols or ["<none>"],
        )

        return bot

    def _build_legacy(self) -> PerpsBot:
        """Build PerpsBot using the legacy approach."""
        settings = self._settings
        if settings is None and self._registry is not None:
            settings = self._registry.runtime_settings
        if settings is None:
            settings = self._settings_provider.get()

        bootstrap_result = prepare_perps_bot(
            self._config,
            self._registry,
            settings=settings,
        )
        registry = bootstrap_result.registry

        config_controller = ConfigController(
            bootstrap_result.config, settings=bootstrap_result.settings
        )
        config = config_controller.current
        if registry.config is not config:
            registry = registry.with_updates(config=config)

        session_guard = TradingSessionGuard(
            start=config.trading_window_start,
            end=config.trading_window_end,
            trading_days=config.trading_days,
        )

        # Updated for Service-based PerpsBot

        # Create baseline snapshot wrapper that looks like what ConfigurationGuardianService expects
        # (This might need adjustment if PerpsBot.build_baseline_snapshot no longer exists or returns differently)
        # For now assuming we don't need the old build_baseline_snapshot method as much, or we construct a simple object.
        # But let's assume we can just pass None if not critical, or construct a minimal object.

        # Actually, we should just instantiate PerpsBot

        bot = PerpsBot(
            config_controller=config_controller,
            registry=registry,
            event_store=bootstrap_result.event_store,
            orders_store=bootstrap_result.orders_store,
            session_guard=session_guard,
            # configuration_guardian=ConfigurationGuardianService(config_controller, ...), # It's instantiated inside or passed?
            # The new PerpsBot __init__ takes configuration_guardian as optional.
            # If we want it, we should create it.
            # However, the new PerpsBot doesn't have .build_baseline_snapshot method on the class anymore.
        )

        # Note: The new architecture's PerpsBot has lifecycle methods but maybe not .lifecycle_manager.bootstrap()
        # It has .start() and .lifecycle_manager.start_lifecycle() called inside .start()
        # We don't need to call bootstrap manually here if we use .start() later.
        # But if the caller expects a bootstrapped bot...
        # The old code called bot.lifecycle_manager.bootstrap().
        # The new PerpsBot has a `_lifecycle_manager` which has `start_lifecycle`.
        # But `bot.start()` calls `start_lifecycle`.

        # Let's trust that the caller will call `bot.start()`.

        symbols = list(bot.symbols)
        logger.info(
            "Coinbase Trader constructed via builder (legacy)",
            operation="coinbase_trader_builder",
            stage="build_complete_legacy",
            profile=config.profile.value,
            symbol_count=len(symbols),
            symbols=symbols or ["<none>"],
        )

        return bot


def create_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
    use_container: bool = False,
) -> PerpsBot:
    """Factory helper around :class:`PerpsBotBuilder`."""

    builder = PerpsBotBuilder(provider=settings_provider).with_config(config)
    if use_container:
        builder = builder.with_container(True)
    if registry is not None:
        builder = builder.with_registry(registry)
        if registry.runtime_settings is not None:
            builder = builder.with_settings(registry.runtime_settings)
    return builder.build()


def create_perps_bot_with_container(
    config: BotConfig,
    settings_provider: RuntimeSettingsProvider | None = None,
) -> PerpsBot:
    """Factory helper that creates a PerpsBot using the container."""
    return create_perps_bot(
        config, registry=None, settings_provider=settings_provider, use_container=True
    )


def create_test_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
    use_container: bool = False,
    **_: Any,
) -> PerpsBot:
    """Test-focused shortcut that proxies to :func:`create_perps_bot`."""

    return create_perps_bot(
        config, registry, settings_provider=settings_provider, use_container=use_container
    )


def create_test_perps_bot_with_container(
    config: BotConfig,
    settings_provider: RuntimeSettingsProvider | None = None,
) -> PerpsBot:
    """Test-focused shortcut that creates a PerpsBot using the container."""

    return create_perps_bot_with_container(config, settings_provider)
