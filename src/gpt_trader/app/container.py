from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.orchestration.config_controller import ConfigController
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker
from gpt_trader.orchestration.service_registry import ServiceRegistry
from gpt_trader.orchestration.trading_bot.bot import TradingBot
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore


def create_brokerage(
    event_store: EventStore,
    market_data: MarketDataService,
    product_catalog: ProductCatalog,
    settings: RuntimeSettings,
) -> tuple[CoinbaseClient | DeterministicBroker, EventStore, MarketDataService, ProductCatalog]:
    """
    Factory function to create the brokerage and verify dependencies.

    Returns DeterministicBroker when PERPS_FORCE_MOCK=1 is set.
    """
    # Check for mock mode FIRST - before credential validation
    if settings.perps_force_mock:
        from gpt_trader.orchestration.deterministic_broker import DeterministicBroker

        return DeterministicBroker(), event_store, market_data, product_catalog

    api_key_name = None
    private_key = None

    # Check for credentials file first
    creds_file = settings.raw_env.get("COINBASE_CREDENTIALS_FILE")
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
        api_key_name = settings.raw_env.get("COINBASE_API_KEY_NAME")
    if not private_key:
        private_key = settings.raw_env.get("COINBASE_PRIVATE_KEY")

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
    def __init__(self, config: BotConfig, settings: RuntimeSettings | None = None):
        self.config = config
        self._settings = settings

        self._config_controller: ConfigController | None = None
        self._broker: CoinbaseClient | DeterministicBroker | None = None
        self._event_store: EventStore | None = None
        self._orders_store: OrdersStore | None = None
        self._market_data_service: MarketDataService | None = None
        self._product_catalog: ProductCatalog | None = None

    @property
    def settings(self) -> RuntimeSettings:
        if self._settings is None:
            self._settings = load_runtime_settings()
        return self._settings

    @property
    def config_controller(self) -> ConfigController:
        if self._config_controller is None:
            self._config_controller = ConfigController(self.config)
        return self._config_controller

    @property
    def event_store(self) -> EventStore:
        if self._event_store is None:
            self._event_store = EventStore()
        return self._event_store

    @property
    def orders_store(self) -> OrdersStore:
        if self._orders_store is None:
            self._orders_store = OrdersStore(storage_path="var/data/orders")
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
                settings=self.settings,
            )
        return self._broker

    def reset_broker(self) -> None:
        self._broker = None

    def reset_config(self) -> None:
        self._config_controller = None

    def create_service_registry(self) -> ServiceRegistry:
        registry = ServiceRegistry(self.config)
        registry = registry.with_updates(
            event_store=self.event_store,
            orders_store=self.orders_store,
            broker=self.broker,
            market_data_service=self.market_data_service,
            product_catalog=self.product_catalog,
            runtime_settings=self.settings,
        )
        return registry

    def create_bot(
        self,
        config_controller: ConfigController | None = None,
        registry: ServiceRegistry | None = None,
        event_store: EventStore | None = None,
        orders_store: OrdersStore | None = None,
        session_guard: Any = None,
        baseline_snapshot: Any = None,
        configuration_guardian: Any = None,
    ) -> TradingBot:

        cc = config_controller or self.config_controller
        reg = registry or self.create_service_registry()
        es = event_store or self.event_store
        os = orders_store or self.orders_store

        return TradingBot(
            config=cc.current,
            container=self,
            registry=reg,  # type: ignore[arg-type]
            event_store=es,  # type: ignore[arg-type]
            orders_store=os,
        )


def create_application_container(
    config: BotConfig, settings: RuntimeSettings | None = None
) -> ApplicationContainer:
    return ApplicationContainer(config, settings)
