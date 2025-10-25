"""Initialization concerns for the Perps bot."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.lifecycle_manager import LifecycleManager
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.session_guard import TradingSessionGuard
from bot_v2.orchestration.strategy_orchestrator import StrategyOrchestrator

from .logging import logger

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.app.container import ApplicationContainer
    from bot_v2.monitoring.configuration_guardian import ConfigurationGuardian
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore


class PerpsBotInitializationMixin:
    """Construction helpers and bootstrap routines for the Perps bot."""

    bot_id: str
    start_time: datetime
    running: bool
    _container: ApplicationContainer | None
    config_controller: ConfigController
    config: BotConfig
    registry: ServiceRegistry
    event_store: EventStore
    orders_store: OrdersStore
    _session_guard: TradingSessionGuard
    baseline_snapshot: Any

    def __init__(
        self,
        *,
        config_controller: ConfigController,
        registry: ServiceRegistry,
        event_store: EventStore,
        orders_store: OrdersStore,
        session_guard: TradingSessionGuard,
        baseline_snapshot: Any,
        configuration_guardian: ConfigurationGuardian | None = None,
        container: ApplicationContainer | None = None,
    ) -> None:
        # Basic attributes
        self.bot_id = "coinbase_trader"
        self.start_time = datetime.now(UTC)
        self.running = False

        # Store container reference if provided
        self._container = container

        # Core dependencies
        self.config_controller = config_controller
        self.config = self.config_controller.current
        self.registry = self._align_registry_with_config(registry)
        self.event_store = event_store
        self.orders_store = orders_store
        self._session_guard = session_guard
        self.baseline_snapshot = baseline_snapshot
        self.configuration_guardian = self._resolve_configuration_guardian(
            configuration_guardian, baseline_snapshot
        )

        # Initialize all components in a single, focused method
        self._initialize_components()

    def _initialize_components(self: PerpsBot) -> None:
        """Initialize all bot components in dependency order."""

        # Initialize runtime state
        self._state = self._initialize_symbols_state()
        self._symbol_processor_override = None

        # Initialize orchestrators and coordinators
        self.strategy_orchestrator = StrategyOrchestrator(self)
        self._setup_coordinator_stack()

        # Initialize lifecycle management
        self.lifecycle_manager = LifecycleManager(self)
        self._initialize_service_placeholders()

    @classmethod
    def from_container(
        cls,
        container: ApplicationContainer,
        *,
        session_guard: TradingSessionGuard | None = None,
        baseline_snapshot: Any | None = None,
        configuration_guardian: ConfigurationGuardian | None = None,
    ) -> PerpsBot:
        """
        Create a PerpsBot instance from an ApplicationContainer.

        This class method provides a convenient way to create a PerpsBot
        using dependencies from an ApplicationContainer. This is the
        recommended way to create PerpsBot instances when using the
        composition root pattern.
        """
        # Get dependencies from container
        config_controller = container.config_controller
        config = config_controller.current
        registry = container.create_service_registry()
        event_store = container.event_store
        orders_store = container.orders_store

        # Create session guard if not provided
        if session_guard is None:
            session_guard = TradingSessionGuard(
                start=config.trading_window_start,
                end=config.trading_window_end,
                trading_days=config.trading_days,
            )

        # Create baseline snapshot if not provided
        if baseline_snapshot is None:
            baseline_snapshot = cls.build_baseline_snapshot(
                config,
                getattr(config, "derivatives_enabled", False),
            )

        # Create configuration guardian if not provided
        if configuration_guardian is None:
            from bot_v2.monitoring.configuration_guardian import (
                ConfigurationGuardian as _ConfigurationGuardian,
            )

            configuration_guardian = _ConfigurationGuardian(baseline_snapshot)

        # Create bot instance
        bot = cls(
            config_controller=config_controller,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            session_guard=session_guard,
            baseline_snapshot=baseline_snapshot,
            configuration_guardian=configuration_guardian,
            container=container,
        )

        # Bootstrap the bot lifecycle
        bot.lifecycle_manager.bootstrap()

        logger.info(
            "Created PerpsBot from container",
            operation="perps_bot_from_container",
            bot_id=bot.bot_id,
            symbol_count=len(bot.symbols),
            symbols=bot.symbols or ["<none>"],
        )

        return bot

    @property
    def container(self) -> ApplicationContainer | None:
        """Get the application container if available."""
        return self._container

    def _align_registry_with_config(self, registry: ServiceRegistry) -> ServiceRegistry:
        """Ensure the registry reflects the runtime configuration."""

        if registry.config is not self.config:
            return registry.with_updates(config=self.config)
        return registry

    def _resolve_configuration_guardian(
        self,
        configuration_guardian: ConfigurationGuardian | None,
        baseline_snapshot: Any,
    ):
        """Return the configuration guardian instance backing drift detection."""

        if configuration_guardian is not None:
            return configuration_guardian

        from bot_v2.monitoring.configuration_guardian import (
            ConfigurationGuardian as _ConfigurationGuardian,
        )

        return _ConfigurationGuardian(baseline_snapshot)

    def _initialize_symbols_state(self: PerpsBot) -> PerpsBotRuntimeState:
        """Initialise symbol configuration and backing runtime state."""

        symbols = list(self.config.symbols or [])
        if not symbols:
            logger.warning(
                "No symbols configured; continuing with empty symbol list",
                operation="coinbase_trader_init",
                stage="symbols_missing",
            )

        self.symbols = symbols
        self._derivatives_enabled = bool(getattr(self.config, "derivatives_enabled", False))
        return PerpsBotRuntimeState(symbols)

    def _initialize_service_placeholders(self) -> None:
        """Reset service placeholders that are populated during telemetry startup."""

        self.account_manager = None
        self.account_telemetry = None
        self.market_monitor = None
        self.intx_portfolio_service = None
