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

    def __init__(self, provider: RuntimeSettingsProvider | None = None) -> None:
        self._config: BotConfig | None = None
        self._registry: ServiceRegistry | None = None
        self._settings: RuntimeSettings | None = None
        self._settings_provider: RuntimeSettingsProvider = (
            provider or DEFAULT_RUNTIME_SETTINGS_PROVIDER
        )

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

    def build(self) -> PerpsBot:
        if self._config is None:
            raise ValueError("Configuration must be supplied before building the bot")

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

        baseline_snapshot = PerpsBot.build_baseline_snapshot(
            config,
            getattr(config, "derivatives_enabled", False),
        )

        from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian

        bot = PerpsBot(
            config_controller=config_controller,
            registry=registry,
            event_store=bootstrap_result.event_store,
            orders_store=bootstrap_result.orders_store,
            session_guard=session_guard,
            baseline_snapshot=baseline_snapshot,
            configuration_guardian=ConfigurationGuardian(baseline_snapshot),
        )

        bot.lifecycle_manager.bootstrap()

        symbols = list(bot.symbols)
        logger.info(
            "Coinbase Trader constructed via builder",
            operation="coinbase_trader_builder",
            stage="build_complete",
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
) -> PerpsBot:
    """Factory helper around :class:`PerpsBotBuilder`."""

    builder = PerpsBotBuilder(provider=settings_provider).with_config(config)
    if registry is not None:
        builder = builder.with_registry(registry)
        if registry.runtime_settings is not None:
            builder = builder.with_settings(registry.runtime_settings)
    return builder.build()


def create_test_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    *,
    settings_provider: RuntimeSettingsProvider | None = None,
    **_: Any,
) -> PerpsBot:
    """Test-focused shortcut that proxies to :func:`create_perps_bot`."""

    return create_perps_bot(config, registry, settings_provider=settings_provider)
