"""Tests for CoinbaseRestServiceCore initialization."""

from __future__ import annotations

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.rest_service_core_test_base import (
    RestServiceCoreTestBase,
)


class TestCoinbaseRestServiceCoreInitialization(RestServiceCoreTestBase):
    def test_service_init(self) -> None:
        assert self.service.client == self.client
        assert self.service.endpoints == self.endpoints
        assert self.service.config == self.config
        assert self.service.product_catalog == self.product_catalog
        assert self.service.market_data == self.market_data
        assert self.service._event_store == self.event_store
        assert isinstance(self.service.positions, dict)
        assert len(self.service.positions) == 0
