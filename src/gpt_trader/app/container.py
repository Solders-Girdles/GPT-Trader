from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.orchestration.config_controller import ConfigController
from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.orchestration.service_registry import ServiceRegistry
from gpt_trader.orchestration.trading_bot.bot import TradingBot
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import OrdersStore


def create_brokerage(
    event_store: EventStore,
    market_data: MarketDataService,
    product_catalog: ProductCatalog,
    settings: RuntimeSettings,
) -> Tuple[CoinbaseClient, EventStore, MarketDataService, ProductCatalog]:
    """
    Factory function to create the brokerage and verify dependencies.
    """
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
    def __init__(self, config: BotConfig, settings: Optional[RuntimeSettings] = None):
        self.config = config
        self._settings = settings

        self._config_controller: Optional[ConfigController] = None
        self._broker: Optional[CoinbaseClient] = None
        self._event_store: Optional[EventStore] = None
        self._orders_store: Optional[OrdersStore] = None
        self._market_data_service: Optional[MarketDataService] = None
        self._product_catalog: Optional[ProductCatalog] = None

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
    def broker(self) -> CoinbaseClient:
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
        config_controller: Optional[ConfigController] = None,
        registry: Optional[ServiceRegistry] = None,
        event_store: Optional[EventStore] = None,
        orders_store: Optional[OrdersStore] = None,
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
            registry=reg,
            event_store=es,
            orders_store=os,
        )


def create_application_container(
    config: BotConfig, settings: Optional[RuntimeSettings] = None
) -> ApplicationContainer:
    return ApplicationContainer(config, settings)
