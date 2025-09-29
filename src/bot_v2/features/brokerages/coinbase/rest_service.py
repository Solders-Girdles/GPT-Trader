"""Composable REST service facade for Coinbase brokerage operations."""

from __future__ import annotations

from ....persistence.event_store import EventStore
from .client import CoinbaseClient
from .endpoints import CoinbaseEndpoints
from .market_data_service import MarketDataService
from .models import APIConfig
from .rest import (
    CoinbaseRestServiceBase,
    OrderRestMixin,
    PnLRestMixin,
    PortfolioRestMixin,
    ProductRestMixin,
)
from .utilities import ProductCatalog


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
