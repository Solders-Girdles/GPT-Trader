"""Builder utilities for constructing :class:`PerpsBot` instances."""

from __future__ import annotations

import logging
from typing import Any

from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.perps_bootstrap import prepare_perps_bot
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard

logger = logging.getLogger(__name__)


class PerpsBotBuilder:
    """Composable builder that assembles a :class:`PerpsBot` without side effects."""

    def __init__(self) -> None:
        self._config: BotConfig | None = None
        self._registry: ServiceRegistry | None = None

    def with_config(self, config: BotConfig) -> PerpsBotBuilder:
        self._config = config
        return self

    def with_registry(self, registry: ServiceRegistry) -> PerpsBotBuilder:
        self._registry = registry
        return self

    def build(self) -> PerpsBot:
        if self._config is None:
            raise ValueError("Configuration must be supplied before building the bot")

        bootstrap_result = prepare_perps_bot(self._config, self._registry)
        registry = bootstrap_result.registry

        config_controller = ConfigController(bootstrap_result.config)
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

        bot = PerpsBot(
            config_controller=config_controller,
            registry=registry,
            event_store=bootstrap_result.event_store,
            orders_store=bootstrap_result.orders_store,
            session_guard=session_guard,
            baseline_snapshot=baseline_snapshot,
            configuration_guardian=ConfigurationGuardian(baseline_snapshot),
        )

        bot._construct_services()
        bot.runtime_coordinator.bootstrap()
        bot._init_accounting_services()
        bot._init_market_services()
        bot._start_streaming_if_configured()

        logger.info(
            "PerpsBot constructed via builder - profile=%s symbols=%s",
            config.profile.value,
            ", ".join(bot.symbols) or "<none>",
        )

        return bot


def create_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
) -> PerpsBot:
    """Factory helper around :class:`PerpsBotBuilder`."""

    builder = PerpsBotBuilder().with_config(config)
    if registry is not None:
        builder = builder.with_registry(registry)
    return builder.build()


def create_test_perps_bot(
    config: BotConfig,
    registry: ServiceRegistry | None = None,
    **_: Any,
) -> PerpsBot:
    """Test-focused shortcut that proxies to :func:`create_perps_bot`."""

    return create_perps_bot(config, registry)
