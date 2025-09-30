"""Composable REST service facade for Coinbase brokerage operations."""

from __future__ import annotations

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.rest import (
    CoinbaseRestServiceBase,
    OrderRestMixin,
    PnLRestMixin,
    PortfolioRestMixin,
    ProductRestMixin,
)
from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
from bot_v2.persistence.event_store import EventStore


class CoinbaseRestService(
    CoinbaseRestServiceBase,
    ProductRestMixin,
    OrderRestMixin,
    PortfolioRestMixin,
    PnLRestMixin,
):
    """Layered orchestration for Coinbase REST endpoints."""

    def __init__(
        self,
        *,
        client: CoinbaseClient,
        endpoints: CoinbaseEndpoints,
        config: APIConfig,
        product_catalog: ProductCatalog,
        market_data: MarketDataService,
        event_store: EventStore,
    ) -> None:
        super().__init__(
            client=client,
            endpoints=endpoints,
            config=config,
            product_catalog=product_catalog,
            market_data=market_data,
            event_store=event_store,
        )


__all__ = ["CoinbaseRestService"]
