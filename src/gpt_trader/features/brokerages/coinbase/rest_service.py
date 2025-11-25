"""Composable REST service facade for Coinbase brokerage operations."""

from __future__ import annotations

from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from gpt_trader.features.brokerages.coinbase.rest.base import CoinbaseRestServiceBase
from gpt_trader.features.brokerages.coinbase.rest.orders import OrderRestMixin
from gpt_trader.features.brokerages.coinbase.rest.pnl import PnLRestMixin
from gpt_trader.features.brokerages.coinbase.rest.portfolio import PortfolioRestMixin
from gpt_trader.features.brokerages.coinbase.rest.product import ProductRestMixin
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.persistence.event_store import EventStore


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
        settings: RuntimeSettings | None = None,
    ) -> None:
        runtime_settings = settings or load_runtime_settings()
        super().__init__(
            client=client,
            endpoints=endpoints,
            config=config,
            product_catalog=product_catalog,
            market_data=market_data,
            event_store=event_store,
            settings=runtime_settings,
        )


__all__ = ["CoinbaseRestService"]
