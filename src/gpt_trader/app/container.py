from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.orchestration.config_controller import ConfigController
from gpt_trader.orchestration.configuration import BotConfig, Profile
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker
from gpt_trader.orchestration.protocols import EventStoreProtocol
from gpt_trader.orchestration.runtime_paths import RuntimePaths, resolve_runtime_paths
from gpt_trader.orchestration.service_registry import ServiceRegistry
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.risk.manager import LiveRiskManager
    from gpt_trader.monitoring.notifications.service import NotificationService
    from gpt_trader.orchestration.trading_bot.bot import TradingBot


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

    api_key_name = None
    private_key = None

    # Check for credentials file first
    creds_file = os.getenv("COINBASE_CREDENTIALS_FILE")
    if creds_file:
        path = Path(creds_file)
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                    api_key_name = data.get("name")
                    private_key = data.get("privateKey")
            except Exception as e:
                raise ValueError(f"Failed to read credentials file {creds_file}: {e}")

    # Fallback to direct env vars
    if not api_key_name:
        api_key_name = os.getenv("COINBASE_API_KEY_NAME") or os.getenv("COINBASE_CDP_API_KEY")
    if not private_key:
        private_key = os.getenv("COINBASE_PRIVATE_KEY") or os.getenv("COINBASE_CDP_PRIVATE_KEY")

    if not api_key_name or not private_key:
        raise ValueError(
            "Coinbase Credentials not found. Set COINBASE_CREDENTIALS_FILE to a JSON key file, "
            "or set COINBASE_API_KEY_NAME and COINBASE_PRIVATE_KEY environment variables."
        )

    auth_client = SimpleAuth(key_name=api_key_name, private_key=private_key)

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

    def reset_broker(self) -> None:
        self._broker = None

    def reset_config(self) -> None:
        self._config_controller = None

    def reset_risk_manager(self) -> None:
        self._risk_manager = None

    def create_service_registry(self) -> ServiceRegistry:
        """
        Create a ServiceRegistry populated with all container services.

        Note: ServiceRegistry is a legacy pattern maintained for backward
        compatibility. New code should access services directly from the
        container.
        """
        registry = ServiceRegistry(self.config)
        registry = registry.with_updates(
            event_store=self.event_store,
            orders_store=self.orders_store,
            broker=self.broker,
            risk_manager=self.risk_manager,
            notification_service=self.notification_service,
            market_data_service=self.market_data_service,
            product_catalog=self.product_catalog,
        )
        return registry

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


def create_application_container(
    config: BotConfig,
) -> ApplicationContainer:
    return ApplicationContainer(config)
