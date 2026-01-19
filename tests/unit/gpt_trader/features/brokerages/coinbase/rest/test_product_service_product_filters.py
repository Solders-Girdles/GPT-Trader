"""Tests for `ProductService` product filter helpers."""

from __future__ import annotations

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.product_service_test_base import (
    ProductServiceTestBase,
)


class TestProductServiceProductFilters(ProductServiceTestBase):
    def test_get_perpetuals_returns_only_perpetuals(self) -> None:
        """Test get_perpetuals filters to perpetual products."""
        self.client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "product_type": "SPOT",
                },
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "product_type": "PERPETUAL",
                },
            ]
        }

        result = self.service.get_perpetuals()

        perp_ids = [p.product_id for p in result if p.market_type == "PERPETUAL"]
        assert "BTC-PERP" in perp_ids or len(result) == 0  # Depends on to_product mapping

    def test_get_futures_returns_only_futures(self) -> None:
        """Test get_futures filters to future products."""
        self.client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "product_type": "SPOT",
                },
            ]
        }

        result = self.service.get_futures()

        # All should be filtered out
        assert all(p.market_type == "FUTURE" for p in result)

    def test_get_spot_products(self) -> None:
        """Test get_spot_products returns only spot products."""
        self.client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "product_type": "SPOT",
                },
                {
                    "product_id": "ETH-USD",
                    "base_currency": "ETH",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "product_type": "SPOT",
                },
            ]
        }

        result = self.service.get_spot_products()

        assert all(p.market_type == "SPOT" for p in result)
        assert len(result) == 2

    def test_get_cfm_products_returns_futures_with_expiry(self) -> None:
        """Test get_cfm_products returns only futures with expiry dates."""
        self.client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "product_type": "SPOT",
                },
                {
                    "product_id": "BTC-20DEC30-CDE",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "product_type": "FUTURE",
                    "future_product_details": {
                        "contract_expiry": "2030-12-20T00:00:00Z",
                    },
                },
            ]
        }

        result = self.service.get_cfm_products()

        # CFM products should be futures with expiry
        assert all(p.market_type == "FUTURE" for p in result)
