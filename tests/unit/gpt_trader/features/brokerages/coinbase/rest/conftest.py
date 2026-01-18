from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig, Product
from gpt_trader.features.brokerages.coinbase.rest.base import CoinbaseRestServiceCore
from gpt_trader.features.brokerages.coinbase.rest.position_state_store import PositionStateStore
from gpt_trader.features.brokerages.coinbase.utilities import ProductCatalog
from gpt_trader.persistence.event_store import EventStore


@dataclass(frozen=True)
class RestServiceCoreHarness:
    service: CoinbaseRestServiceCore
    client: Mock
    endpoints: Mock
    config: Mock
    product_catalog: Mock
    market_data: Mock
    event_store: Mock
    position_store: PositionStateStore
    mock_product: Mock


@pytest.fixture
def rest_service_core_harness() -> RestServiceCoreHarness:
    client = Mock(spec=CoinbaseClient)
    endpoints = Mock(spec=CoinbaseEndpoints)
    config = Mock(spec=APIConfig)
    product_catalog = Mock(spec=ProductCatalog)
    market_data = Mock(spec=MarketDataService)
    event_store = Mock(spec=EventStore)

    position_store = PositionStateStore()
    service = CoinbaseRestServiceCore(
        client=client,
        endpoints=endpoints,
        config=config,
        product_catalog=product_catalog,
        market_data=market_data,
        event_store=event_store,
        position_store=position_store,
    )

    mock_product = Mock(spec=Product)
    mock_product.product_id = "BTC-USD"
    mock_product.step_size = Decimal("0.00000001")
    mock_product.price_increment = Decimal("0.01")
    mock_product.min_size = Decimal("0.001")
    mock_product.min_notional = Decimal("10")

    return RestServiceCoreHarness(
        service=service,
        client=client,
        endpoints=endpoints,
        config=config,
        product_catalog=product_catalog,
        market_data=market_data,
        event_store=event_store,
        position_store=position_store,
        mock_product=mock_product,
    )
