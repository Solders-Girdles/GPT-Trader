"""Shared fixtures for Coinbase REST contract suite tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.endpoints import CoinbaseEndpoints
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.models import APIConfig, Product
from gpt_trader.features.brokerages.coinbase.rest.base import CoinbaseRestServiceCore
from gpt_trader.features.brokerages.coinbase.rest.order_service import OrderService
from gpt_trader.features.brokerages.coinbase.rest.pnl_service import PnLService
from gpt_trader.features.brokerages.coinbase.rest.portfolio_service import PortfolioService
from gpt_trader.features.brokerages.coinbase.rest.position_state_store import PositionStateStore
from gpt_trader.persistence.event_store import EventStore


class CoinbaseRestContractSuiteBase:
    """Contract fixtures for Coinbase REST service components."""

    @pytest.fixture
    def mock_client(self) -> Mock:
        return Mock(spec=CoinbaseClient)

    @pytest.fixture
    def mock_endpoints(self) -> Mock:
        return Mock(spec=CoinbaseEndpoints)

    @pytest.fixture
    def mock_config(self) -> Mock:
        return Mock(spec=APIConfig)

    @pytest.fixture
    def mock_product_catalog(self) -> Mock:
        return Mock()

    @pytest.fixture
    def mock_market_data(self) -> Mock:
        return Mock(spec=MarketDataService)

    @pytest.fixture
    def mock_event_store(self) -> Mock:
        return Mock(spec=EventStore)

    @pytest.fixture
    def mock_product(self) -> Mock:
        product = Mock(spec=Product)
        product.product_id = "BTC-USD"
        product.step_size = Decimal("0.00000001")
        product.price_increment = Decimal("0.01")
        product.min_size = Decimal("0.001")
        product.min_notional = Decimal("10")
        return product

    @pytest.fixture
    def position_store(self) -> PositionStateStore:
        return PositionStateStore()

    @pytest.fixture
    def service_core(
        self,
        mock_client,
        mock_endpoints,
        mock_config,
        mock_product_catalog,
        mock_market_data,
        mock_event_store,
        position_store,
    ) -> CoinbaseRestServiceCore:
        return CoinbaseRestServiceCore(
            client=mock_client,
            endpoints=mock_endpoints,
            config=mock_config,
            product_catalog=mock_product_catalog,
            market_data=mock_market_data,
            event_store=mock_event_store,
            position_store=position_store,
        )

    @pytest.fixture
    def portfolio_service(self, mock_client, mock_endpoints, mock_event_store) -> PortfolioService:
        return PortfolioService(
            client=mock_client,
            endpoints=mock_endpoints,
            event_store=mock_event_store,
        )

    @pytest.fixture
    def order_service(self, mock_client, service_core, portfolio_service) -> OrderService:
        return OrderService(
            client=mock_client,
            payload_builder=service_core,
            payload_executor=service_core,
            position_provider=portfolio_service,
        )

    @pytest.fixture
    def pnl_service(self, position_store, mock_market_data) -> PnLService:
        return PnLService(
            position_store=position_store,
            market_data=mock_market_data,
        )
