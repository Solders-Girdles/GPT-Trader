"""
Application composition root with dependency injection container.

This module implements the composition root pattern with a lightweight dependency
injection container to manage application dependencies. The container is responsible
for creating and configuring the core services that the PerpsBot depends on,
particularly configuration and broker services.

The container follows these principles:
- Single responsibility: Each factory method creates one type of service
- Lazy initialization: Services are created only when needed
- Dependency injection: Services receive their dependencies through constructors
- Configuration-driven: Service creation is based on configuration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings

from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.features.brokerages.core.interfaces import IBrokerage
from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.orchestration.config_controller import ConfigController
from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.runtime_settings import load_runtime_settings
from bot_v2.orchestration.state.unified_state import (
    SystemState,
    create_reduce_only_state_manager,
)
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="application_container")

PerpsBot: Any | None = None
ConfigurationGuardian: Any | None = None


class ApplicationContainer:
    """
    Dependency injection container for application services.

    The container manages the lifecycle and dependencies of core application services.
    It provides a centralized place for service creation and configuration,
    making the application more testable and maintainable.
    """

    def __init__(self, config: BotConfig, settings: RuntimeSettings | None = None) -> None:
        """
        Initialize the container with configuration.

        Args:
            config: The bot configuration
            settings: Runtime settings (optional, will be loaded if not provided)
        """
        self._config = config
        self._settings = settings
        self._config_controller: ConfigController | None = None
        self._broker: IBrokerage | None = None
        self._event_store: EventStore | None = None
        self._orders_store: OrdersStore | None = None
        self._market_data_service: MarketDataService | None = None
        self._product_catalog: ProductCatalog | None = None
        self._reduce_only_state_manager: SystemState | None = None

    @property
    def config(self) -> BotConfig:
        """Get the bot configuration."""
        return self._config

    @property
    def settings(self) -> RuntimeSettings:
        """Get runtime settings, loading if necessary."""
        if self._settings is None:
            self._settings = load_runtime_settings()
        return self._settings

    @property
    def config_controller(self) -> ConfigController:
        """
        Get the configuration controller, creating if necessary.

        The config controller manages configuration lifecycle and synchronization
        with runtime state.
        """
        if self._config_controller is None:
            # Ensure state manager is created
            state_manager = self.reduce_only_state_manager

            self._config_controller = ConfigController(
                self._config, settings=self.settings, reduce_only_state_manager=state_manager
            )
            logger.debug(
                "Created config controller",
                operation="container",
                service="config_controller",
            )
        return self._config_controller

    @property
    def broker(self) -> IBrokerage:
        """
        Get the broker instance, creating if necessary.

        The broker is created based on configuration and environment variables.
        It includes all necessary dependencies like event store, market data service,
        and product catalog.
        """
        if self._broker is None:
            # Ensure downstream factories receive fully initialized dependencies
            event_store = self.event_store
            market_data = self.market_data_service
            product_catalog = self.product_catalog

            self._broker, event_store, market_data, product_catalog = create_brokerage(
                event_store=event_store,
                market_data=market_data,
                product_catalog=product_catalog,
                settings=self.settings,
            )

            # Store the created instances to ensure singleton behavior
            self._event_store = event_store
            self._market_data_service = market_data
            self._product_catalog = product_catalog

            logger.debug(
                "Created broker",
                operation="container",
                service="broker",
                broker_type=type(self._broker).__name__,
            )
        return self._broker

    @property
    def event_store(self) -> EventStore:
        """
        Get the event store, creating if necessary.

        The event store is used for persisting trading events and state changes.
        """
        if self._event_store is None:
            self._event_store = EventStore()
            logger.debug(
                "Created event store",
                operation="container",
                service="event_store",
            )
        return self._event_store

    @property
    def orders_store(self) -> OrdersStore:
        """
        Get the orders store, creating if necessary.

        The orders store is used for persisting order information and state.
        """
        if self._orders_store is None:
            # Use a default path based on settings
            storage_path = self.settings.runtime_root / "orders"
            self._orders_store = OrdersStore(storage_path=storage_path)
            logger.debug(
                "Created orders store",
                operation="container",
                service="orders_store",
                storage_path=str(storage_path),
            )
        return self._orders_store

    @property
    def market_data_service(self) -> MarketDataService:
        """
        Get the market data service, creating if necessary.

        The market data service provides real-time market data for trading decisions.
        """
        if self._market_data_service is None:
            self._market_data_service = MarketDataService()
            logger.debug(
                "Created market data service",
                operation="container",
                service="market_data_service",
            )
        return self._market_data_service

    @property
    def product_catalog(self) -> ProductCatalog:
        """
        Get the product catalog, creating if necessary.

        The product catalog provides information about available trading products.
        """
        if self._product_catalog is None:
            # Use a reasonable TTL for product information
            self._product_catalog = ProductCatalog(ttl_seconds=900)
            logger.debug(
                "Created product catalog",
                operation="container",
                service="product_catalog",
                ttl_seconds=900,
            )
        return self._product_catalog

    @property
    def reduce_only_state_manager(self) -> SystemState:
        """
        Get the reduce-only state manager, creating if necessary.

        The state manager provides centralized control over reduce_only_mode
        mutations with validation and audit logging.
        """
        if self._reduce_only_state_manager is None:
            # Ensure event store is created
            _ = self.event_store

            # Get initial state from config
            initial_state = bool(getattr(self._config, "reduce_only_mode", False))

            self._reduce_only_state_manager = create_reduce_only_state_manager(
                event_store=self._event_store,
                initial_state=initial_state,
                validation_enabled=True,
            )

            logger.debug(
                "Created reduce-only state manager",
                operation="container",
                service="reduce_only_state_manager",
                initial_state=initial_state,
            )
        return self._reduce_only_state_manager

    def create_service_registry(self) -> Any:
        """
        Create a service registry populated with container services.

        This method creates a ServiceRegistry instance populated with the
        services managed by this container. This provides backward compatibility
        with existing code that expects a ServiceRegistry.

        Returns:
            ServiceRegistry populated with container services
        """
        from bot_v2.orchestration.service_registry import ServiceRegistry

        registry = ServiceRegistry(
            config=self.config_controller.current,
            broker=self.broker,
            event_store=self.event_store,
            orders_store=self.orders_store,
            market_data_service=self.market_data_service,
            product_catalog=self.product_catalog,
            runtime_settings=self.settings,
            reduce_only_state_manager=self.reduce_only_state_manager,
        )

        # If risk manager is already created, inject the state manager
        if registry.risk_manager is not None and self.reduce_only_state_manager is not None:
            registry.risk_manager._centralized_state_manager = self.reduce_only_state_manager

        logger.debug(
            "Created service registry",
            operation="container",
            service="service_registry",
        )

        return registry

    def create_perps_bot(self, **overrides: Any) -> Any:
        """
        Create a PerpsBot instance with dependencies from this container.

        This method creates a PerpsBot instance using the services managed
        by this container. It provides a convenient way to create a fully
        configured bot instance.

        Args:
            **overrides: Optional overrides for bot dependencies

        Returns:
            Configured PerpsBot instance
        """
        from bot_v2.orchestration.session_guard import TradingSessionGuard

        perps_bot_cls = globals().get("PerpsBot")
        if perps_bot_cls is None:
            from bot_v2.orchestration.perps_bot import PerpsBot as _PerpsBot

            perps_bot_cls = _PerpsBot
            globals()["PerpsBot"] = _PerpsBot

        # Get dependencies from container
        config = self.config_controller.current
        registry = self.create_service_registry()
        event_store = self.event_store
        orders_store = self.orders_store

        # Create session guard
        session_guard = TradingSessionGuard(
            start=config.trading_window_start,
            end=config.trading_window_end,
            trading_days=config.trading_days,
        )

        # Create baseline snapshot for configuration drift detection
        baseline_snapshot = perps_bot_cls.build_baseline_snapshot(
            config,
            getattr(config, "derivatives_enabled", False),
        )

        # Apply any overrides
        config_controller = overrides.get("config_controller", self.config_controller)
        registry = overrides.get("registry", registry)
        event_store = overrides.get("event_store", event_store)
        orders_store = overrides.get("orders_store", orders_store)
        session_guard = overrides.get("session_guard", session_guard)
        baseline_snapshot = overrides.get("baseline_snapshot", baseline_snapshot)

        # Create configuration guardian
        configuration_guardian_cls = globals().get("ConfigurationGuardian")
        if configuration_guardian_cls is None:
            from bot_v2.monitoring.configuration_guardian import (
                ConfigurationGuardian as _ConfigurationGuardian,
            )

            configuration_guardian_cls = _ConfigurationGuardian
            globals()["ConfigurationGuardian"] = _ConfigurationGuardian

        configuration_guardian = overrides.get(
            "configuration_guardian",
            configuration_guardian_cls(baseline_snapshot),
        )

        bot = perps_bot_cls(
            config_controller=config_controller,
            registry=registry,
            event_store=event_store,
            orders_store=orders_store,
            session_guard=session_guard,
            baseline_snapshot=baseline_snapshot,
            configuration_guardian=configuration_guardian,
            container=self,
        )

        # Ensure the StateManager is properly injected into the risk manager
        if registry.reduce_only_state_manager is not None and registry.risk_manager is not None:
            registry.risk_manager._centralized_state_manager = registry.reduce_only_state_manager

        # Bootstrap the bot lifecycle
        bot.lifecycle_manager.bootstrap()

        logger.info(
            "Created PerpsBot from container",
            operation="container",
            service="perps_bot",
            bot_id=bot.bot_id,
            symbol_count=len(bot.symbols),
            symbols=bot.symbols or ["<none>"],
        )

        return bot

    def reset_broker(self) -> None:
        """
        Reset the broker instance.

        This method forces the recreation of the broker on next access.
        Useful for testing or when broker configuration changes.
        """
        self._broker = None
        logger.debug(
            "Reset broker instance",
            operation="container",
            action="reset_broker",
        )

    def reset_config(self) -> None:
        """
        Reset the config controller instance.

        This method forces the recreation of the config controller on next access.
        Useful when configuration needs to be reloaded.
        """
        self._config_controller = None
        logger.debug(
            "Reset config controller instance",
            operation="container",
            action="reset_config",
        )

    def reset_reduce_only_state_manager(self) -> None:
        """
        Reset the reduce-only state manager instance.

        This method forces the recreation of the state manager on next access.
        Useful when state management needs to be reset.
        """
        self._reduce_only_state_manager = None
        logger.debug(
            "Reset reduce-only state manager instance",
            operation="container",
            action="reset_reduce_only_state_manager",
        )


def create_application_container(
    config: BotConfig, settings: RuntimeSettings | None = None
) -> ApplicationContainer:
    """
    Create an application container with the given configuration.

    This is a convenience function that creates an ApplicationContainer
    instance with the provided configuration and optional settings.

    Args:
        config: The bot configuration
        settings: Runtime settings (optional)

    Returns:
        Configured ApplicationContainer instance
    """
    container = ApplicationContainer(config, settings)
    logger.debug(
        "Created application container",
        operation="container_factory",
        profile=config.profile.value,
    )
    return container


__all__ = [
    "ApplicationContainer",
    "create_application_container",
    "PerpsBot",
    "ConfigurationGuardian",
]


def __getattr__(name: str):
    if name == "PerpsBot":
        from bot_v2.orchestration.perps_bot import PerpsBot as _PerpsBot

        globals()["PerpsBot"] = _PerpsBot
        return _PerpsBot
    if name == "ConfigurationGuardian":
        from bot_v2.monitoring.configuration_guardian import (
            ConfigurationGuardian as _ConfigurationGuardian,
        )

        globals()["ConfigurationGuardian"] = _ConfigurationGuardian
        return _ConfigurationGuardian
    raise AttributeError(name)
