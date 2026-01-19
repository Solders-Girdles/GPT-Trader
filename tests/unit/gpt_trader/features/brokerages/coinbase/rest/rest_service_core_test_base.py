from __future__ import annotations

import pytest

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.conftest import (
    RestServiceCoreHarness,
)


class RestServiceCoreTestBase:
    @pytest.fixture(autouse=True)
    def _setup_rest_service_core(self, rest_service_core_harness: RestServiceCoreHarness) -> None:
        self.client = rest_service_core_harness.client
        self.endpoints = rest_service_core_harness.endpoints
        self.config = rest_service_core_harness.config
        self.product_catalog = rest_service_core_harness.product_catalog
        self.market_data = rest_service_core_harness.market_data
        self.event_store = rest_service_core_harness.event_store
        self.position_store = rest_service_core_harness.position_store
        self.service = rest_service_core_harness.service
        self.mock_product = rest_service_core_harness.mock_product
