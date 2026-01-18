"""Shared setup for `ProductService` tests."""

from __future__ import annotations

from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.rest.product_service import ProductService


class ProductServiceTestBase:
    def setup_method(self) -> None:
        self.client = Mock()  # Don't use spec to allow dynamic method mocking
        self.product_catalog = Mock()  # Don't use spec
        self.market_data = Mock(spec=MarketDataService)

        self.service = ProductService(
            client=self.client,
            product_catalog=self.product_catalog,
            market_data=self.market_data,
        )
