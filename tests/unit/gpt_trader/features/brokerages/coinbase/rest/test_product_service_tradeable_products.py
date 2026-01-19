"""Tests for `ProductService.get_tradeable_products`."""

from __future__ import annotations

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.product_service_test_base import (
    ProductServiceTestBase,
)


class TestProductServiceTradeableProducts(ProductServiceTestBase):
    def test_get_tradeable_products_spot_only(self) -> None:
        """Test get_tradeable_products with spot mode only."""
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

        result = self.service.get_tradeable_products(["spot"])

        assert len(result) == 1
        assert result[0].symbol == "BTC-USD"
        assert all(p.market_type == "SPOT" for p in result)

    def test_get_tradeable_products_cfm_only(self) -> None:
        """Test get_tradeable_products with CFM mode only."""
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

        result = self.service.get_tradeable_products(["cfm"])

        # Should only include CFM products (futures with expiry)
        assert all(p.market_type == "FUTURE" for p in result)

    def test_get_tradeable_products_hybrid_mode(self) -> None:
        """Test get_tradeable_products with hybrid mode (spot + cfm)."""
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

        result = self.service.get_tradeable_products(["spot", "cfm"])

        # Should include both spot and CFM products (no duplicates)
        symbols = [p.symbol for p in result]
        assert len(symbols) == len(set(symbols))  # No duplicates

    def test_get_tradeable_products_empty_modes(self) -> None:
        """Test get_tradeable_products with empty modes returns nothing."""
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

        result = self.service.get_tradeable_products([])

        assert result == []
