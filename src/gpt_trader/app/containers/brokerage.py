"""Brokerage sub-container for ApplicationContainer.

This container manages brokerage-related dependencies:
- Broker client (CoinbaseClient or DeterministicBroker)
- Market data service
- Product catalog
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from gpt_trader.app.config import BotConfig
from gpt_trader.features.brokerages.coinbase.client.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.orchestration.deterministic_broker import DeterministicBroker
from gpt_trader.persistence.event_store import EventStore

if TYPE_CHECKING:
    pass

# Type alias for the broker factory function signature
BrokerType = CoinbaseClient | DeterministicBroker
BrokerFactoryResult = tuple[BrokerType, EventStore, MarketDataService, ProductCatalog]
BrokerFactory = Callable[
    [EventStore, MarketDataService, ProductCatalog, BotConfig],
    BrokerFactoryResult,
]


class BrokerageContainer:
    """Container for brokerage-related dependencies.

    This container lazily initializes broker, market data service, and product
    catalog. It accepts factory functions to avoid circular imports with the
    main ApplicationContainer.

    Args:
        config: Bot configuration.
        event_store_provider: Callable that returns the EventStore instance.
            This is a callable (not the instance) to support lazy resolution
            and avoid initialization order issues.
        broker_factory: Factory function to create the broker. Signature:
            (event_store, market_data, product_catalog, config) -> (broker, ...)
    """

    def __init__(
        self,
        config: BotConfig,
        event_store_provider: Callable[[], EventStore],
        broker_factory: BrokerFactory,
    ):
        self._config = config
        self._event_store_provider = event_store_provider
        self._broker_factory = broker_factory

        self._broker: BrokerType | None = None
        self._market_data_service: MarketDataService | None = None
        self._product_catalog: ProductCatalog | None = None

    @property
    def market_data_service(self) -> MarketDataService:
        """Get or create the market data service."""
        if self._market_data_service is None:
            self._market_data_service = MarketDataService(symbols=list(self._config.symbols))
        return self._market_data_service

    @property
    def product_catalog(self) -> ProductCatalog:
        """Get or create the product catalog."""
        if self._product_catalog is None:
            self._product_catalog = ProductCatalog()
        return self._product_catalog

    @property
    def broker(self) -> BrokerType:
        """Get or create the broker client.

        Uses the injected broker_factory to create the broker, which handles
        credential resolution and mock mode detection.
        """
        if self._broker is None:
            self._broker, _, _, _ = self._broker_factory(
                self._event_store_provider(),
                self.market_data_service,
                self.product_catalog,
                self._config,
            )
        return self._broker

    def reset_broker(self) -> None:
        """Reset the broker instance, forcing re-creation on next access."""
        self._broker = None
