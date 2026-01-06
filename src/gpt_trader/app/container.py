from __future__ import annotations

from typing import TYPE_CHECKING, cast

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.credentials import resolve_coinbase_credentials
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.orchestration.config_controller import ConfigController
from gpt_trader.orchestration.configuration import BotConfig, Profile
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker
from gpt_trader.orchestration.protocols import EventStoreProtocol
from gpt_trader.orchestration.runtime_paths import RuntimePaths, resolve_runtime_paths
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
    from gpt_trader.monitoring.notifications.service import NotificationService
    from gpt_trader.orchestration.configuration.profile_loader import ProfileLoader
    from gpt_trader.orchestration.execution.validation import ValidationFailureTracker
    from gpt_trader.orchestration.live_execution import LiveExecutionEngine
    from gpt_trader.orchestration.trading_bot.bot import TradingBot


logger = get_logger(__name__, component="container")


def create_brokerage(
    event_store: EventStore,
    market_data: MarketDataService,
    product_catalog: ProductCatalog,
    config: BotConfig,
) -> tuple[CoinbaseClient | DeterministicBroker, EventStore, MarketDataService, ProductCatalog]:
    """
    Factory function to create the brokerage and verify dependencies.

    Returns DeterministicBroker when mock_broker=True.
    """
    # Check for mock mode FIRST - before credential validation
    if config.mock_broker:
        from gpt_trader.orchestration.deterministic_broker import DeterministicBroker

        return DeterministicBroker(), event_store, market_data, product_catalog

    creds = resolve_coinbase_credentials()
    if not creds:
        raise ValueError(
            "Coinbase Credentials not found. Set COINBASE_CREDENTIALS_FILE to a JSON key file, "
            "or set COINBASE_CDP_API_KEY + COINBASE_CDP_PRIVATE_KEY "
            "(legacy: COINBASE_API_KEY_NAME + COINBASE_PRIVATE_KEY)."
        )

    for warning in creds.warnings:
        logger.warning("Coinbase credential configuration: %s", warning)
    logger.info("Using Coinbase credentials from %s (%s)", creds.source, creds.masked_key_name)

    auth_client = SimpleAuth(key_name=creds.key_name, private_key=creds.private_key)

    broker = CoinbaseClient(
        auth=auth_client,
    )
    return broker, event_store, market_data, product_catalog


class ApplicationContainer:
    """
    Canonical composition root for GPT-Trader application.

    This container lazily initializes all application services and provides
    the single source of truth for dependency injection. All services should
    be accessed through this container.

    Usage:
        container = ApplicationContainer(config)
        bot = container.create_bot()  # Creates fully-wired TradingBot
    """

    def __init__(self, config: BotConfig):
        self.config = config

        self._config_controller: ConfigController | None = None
        self._runtime_paths: RuntimePaths | None = None
        self._broker: CoinbaseClient | DeterministicBroker | None = None
        self._event_store: EventStore | None = None
        self._orders_store: OrdersStore | None = None
        self._market_data_service: MarketDataService | None = None
        self._product_catalog: ProductCatalog | None = None
        self._risk_manager: LiveRiskManager | None = None
        self._notification_service: NotificationService | None = None
        self._validation_failure_tracker: ValidationFailureTracker | None = None
        self._profile_loader: ProfileLoader | None = None

    @property
    def config_controller(self) -> ConfigController:
        if self._config_controller is None:
            self._config_controller = ConfigController(self.config)
        return self._config_controller

    @property
    def runtime_paths(self) -> RuntimePaths:
        """Resolve storage directories for the configured profile."""
        if self._runtime_paths is None:
            profile = cast(Profile, self.config.profile)
            self._runtime_paths = resolve_runtime_paths(config=self.config, profile=profile)
        return self._runtime_paths

    @property
    def event_store(self) -> EventStore:
        if self._event_store is None:
            self._event_store = EventStore(root=self.runtime_paths.event_store_root)
        return self._event_store

    @property
    def orders_store(self) -> OrdersStore:
        if self._orders_store is None:
            self._orders_store = OrdersStore(storage_path=self.runtime_paths.storage_dir)
        return self._orders_store

    @property
    def market_data_service(self) -> MarketDataService:
        if self._market_data_service is None:
            self._market_data_service = MarketDataService(symbols=list(self.config.symbols))
        return self._market_data_service

    @property
    def product_catalog(self) -> ProductCatalog:
        if self._product_catalog is None:
            self._product_catalog = ProductCatalog()
        return self._product_catalog

    @property
    def broker(self) -> CoinbaseClient | DeterministicBroker:
        if self._broker is None:
            self._broker, _, _, _ = create_brokerage(
                event_store=self.event_store,
                market_data=self.market_data_service,
                product_catalog=self.product_catalog,
                config=self.config,
            )
        return self._broker

    @property
    def risk_manager(self) -> LiveRiskManager:
        """Create or return the risk manager instance."""
        if self._risk_manager is None:
            from decimal import Decimal

            from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
            from gpt_trader.orchestration.configuration.risk.model import RiskConfig

            # Adapt BotConfig.risk (BotRiskConfig) to RiskConfig
            bot_risk = self.config.risk
            risk_config = RiskConfig(
                max_leverage=bot_risk.max_leverage,
                daily_loss_limit=bot_risk.max_drawdown_pct or Decimal("0.1"),
                max_position_pct_per_symbol=float(bot_risk.position_fraction),
                # Map other relevant fields if needed, or rely on defaults
                kill_switch_enabled=self.config.mean_reversion.kill_switch_enabled,
                reduce_only_mode=self.config.reduce_only_mode,
            )

            self._risk_manager = LiveRiskManager(
                config=risk_config,
                event_store=self.event_store,
            )
        return self._risk_manager

    @property
    def notification_service(self) -> NotificationService:
        """Create or return the notification service instance."""
        if self._notification_service is None:
            from gpt_trader.monitoring.notifications import create_notification_service

            self._notification_service = create_notification_service(
                webhook_url=self.config.webhook_url,
                console_enabled=True,
            )
        return self._notification_service

    @property
    def validation_failure_tracker(self) -> ValidationFailureTracker:
        """Create or return the validation failure tracker instance.

        This tracker monitors consecutive validation failures and can trigger
        escalation (e.g., reduce-only mode) when thresholds are exceeded.

        Note: Default configuration matches the current global tracker behavior
        (threshold=5, no escalation callback). Escalation callback can be
        configured after resolution if needed.
        """
        if self._validation_failure_tracker is None:
            from gpt_trader.orchestration.execution.validation import (
                ValidationFailureTracker as VFT,
            )

            self._validation_failure_tracker = VFT()
        return self._validation_failure_tracker

    @property
    def profile_loader(self) -> ProfileLoader:
        """Create or return the profile loader instance.

        The profile loader handles loading and validating trading profile
        configurations from YAML files.
        """
        if self._profile_loader is None:
            from gpt_trader.orchestration.configuration.profile_loader import (
                ProfileLoader as PL,
            )

            self._profile_loader = PL()
        return self._profile_loader

    def reset_broker(self) -> None:
        self._broker = None

    def reset_config(self) -> None:
        self._config_controller = None

    def reset_risk_manager(self) -> None:
        self._risk_manager = None

    def create_bot(self) -> TradingBot:
        """
        Create a fully-wired TradingBot instance.

        This is the canonical way to create a TradingBot. The container
        provides all necessary dependencies.

        Returns:
            TradingBot: A fully configured trading bot ready to run.
        """
        from gpt_trader.orchestration.trading_bot.bot import TradingBot

        return TradingBot(
            config=self.config,
            container=self,
            event_store=cast(EventStoreProtocol, self.event_store),
            orders_store=self.orders_store,
            notification_service=self.notification_service,
        )

    def create_live_execution_engine(
        self,
        *,
        bot_id: str = "live_execution",
        slippage_multipliers: dict[str, float] | None = None,
        enable_preview: bool | None = None,
        failure_tracker: ValidationFailureTracker | None = None,
    ) -> LiveExecutionEngine:
        """Create a LiveExecutionEngine wired with container dependencies.

        This factory method creates a LiveExecutionEngine using the container's
        broker, risk_manager, event_store, and config. The engine will create
        its own per-instance ValidationFailureTracker with escalation callback
        unless one is explicitly provided.

        Args:
            bot_id: Bot identifier for logging.
            slippage_multipliers: Optional symbol-specific slippage multipliers.
            enable_preview: Enable order preview (defaults to config setting).
            failure_tracker: Optional pre-configured failure tracker. If not
                provided, the engine creates its own with escalation callback.

        Returns:
            LiveExecutionEngine: A configured execution engine ready for use.
        """
        from gpt_trader.orchestration.live_execution import (
            LiveExecutionEngine as LEE,
        )

        return LEE(
            broker=self.broker,
            config=self.config,
            risk_manager=self.risk_manager,
            event_store=self.event_store,
            bot_id=bot_id,
            slippage_multipliers=slippage_multipliers,
            enable_preview=enable_preview,
            failure_tracker=failure_tracker,
        )


def create_application_container(
    config: BotConfig,
) -> ApplicationContainer:
    return ApplicationContainer(config)


# ---------------------------------------------------------------------------
# Module-level container registry
# ---------------------------------------------------------------------------
# Allows services to resolve dependencies via get_application_container()
# without requiring explicit container passing. The container must be set
# during application startup via set_application_container().

_current_container: ApplicationContainer | None = None


def set_application_container(container: ApplicationContainer | None) -> None:
    """Set the current application container for global access.

    Call this during application startup after creating the container.
    Pass None to clear the container (e.g., during shutdown or tests).

    Args:
        container: The ApplicationContainer instance, or None to clear.
    """
    global _current_container
    if container is not None:
        config = getattr(container, "config", None)
        profile = getattr(config, "profile", None) if config else None
        logger.debug("Application container set", profile=profile)
    else:
        logger.debug("Application container cleared via set_application_container(None)")
    _current_container = container


def get_application_container() -> ApplicationContainer | None:
    """Get the current application container.

    Returns:
        The current ApplicationContainer if set, None otherwise.

    Note:
        Returns None if no container has been set. Callers should handle
        this case gracefully (e.g., fall back to creating services directly).
    """
    return _current_container


def clear_application_container() -> None:
    """Clear the current application container.

    Convenience function equivalent to set_application_container(None).
    Useful in tests to ensure clean state between test cases.
    """
    global _current_container
    was_set = _current_container is not None
    _current_container = None
    if was_set:
        logger.debug("Application container cleared")
