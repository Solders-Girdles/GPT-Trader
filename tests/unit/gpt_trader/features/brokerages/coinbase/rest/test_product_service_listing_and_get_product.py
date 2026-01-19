"""Tests for `ProductService` listing and `get_product` behavior."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.core import Product
from tests.unit.gpt_trader.features.brokerages.coinbase.rest.product_service_test_base import (
    ProductServiceTestBase,
)


class TestProductServiceListingAndGetProduct(ProductServiceTestBase):
    """Covers product listing and product retrieval behaviors."""

    def test_service_init(self) -> None:
        """Test service initialization."""
        assert self.service._client == self.client
        assert self.service._product_catalog == self.product_catalog
        assert self.service._market_data == self.market_data

    def test_list_products_returns_products(self) -> None:
        """Test listing products returns parsed products."""
        self.client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "base_min_size": "0.001",
                },
                {
                    "product_id": "ETH-USD",
                    "base_currency": "ETH",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "base_min_size": "0.001",
                },
            ]
        }

        result = self.service.list_products()

        assert len(result) == 2
        assert result[0].symbol == "BTC-USD"
        assert result[1].symbol == "ETH-USD"

    def test_list_products_handles_list_response(self) -> None:
        """Test list_products handles list response shape."""
        self.client.get_products.return_value = [
            {
                "product_id": "BTC-USD",
                "base_currency": "BTC",
                "quote_currency": "USD",
                "status": "online",
                "base_increment": "0.00000001",
                "quote_increment": "0.01",
                "base_min_size": "0.001",
            }
        ]

        result = self.service.list_products()

        assert len(result) == 1

    def test_list_products_skips_invalid_entries(self) -> None:
        """Test list_products skips entries that fail to parse."""
        self.client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "status": "online",
                    "base_increment": "0.00000001",
                    "quote_increment": "0.01",
                    "base_min_size": "0.001",
                },
                {},  # Invalid entry - missing required fields
            ]
        }

        result = self.service.list_products()

        # Should return valid entries only
        assert len(result) >= 1

    def test_list_products_handles_network_error(self) -> None:
        """Test list_products handles network errors."""
        self.client.get_products.side_effect = ConnectionError("Network error")

        result = self.service.list_products()

        assert result == []

    def test_list_products_handles_timeout(self) -> None:
        """Test list_products handles timeout."""
        self.client.get_products.side_effect = TimeoutError("Timeout")

        result = self.service.list_products()

        assert result == []

    def test_list_products_handles_unexpected_error(self) -> None:
        """Test list_products handles unexpected errors."""
        self.client.get_products.side_effect = Exception("Unexpected")

        result = self.service.list_products()

        assert result == []

    def test_get_product_from_catalog(self) -> None:
        """Test getting product from catalog."""
        # Create a simple mock without spec so hasattr works naturally
        mock_product = Mock()
        mock_product.symbol = "BTC-USD"
        mock_product.funding_rate = None
        mock_product.next_funding_time = None

        # Use a SimpleNamespace-like object for catalog that doesn't have get_funding
        from unittest.mock import MagicMock

        catalog_mock = MagicMock()
        catalog_mock.get.return_value = mock_product
        # Explicitly remove get_funding so hasattr returns False
        if hasattr(catalog_mock, "get_funding"):
            del catalog_mock.get_funding

        self.service._product_catalog = catalog_mock

        result = self.service.get_product("BTC-USD")

        assert result is not None
        # Result should be the mock product returned from catalog
        catalog_mock.get.assert_called()

    def test_get_product_with_funding_enrichment(self) -> None:
        """Test get_product enriches with funding data."""
        mock_product = Mock(spec=Product)
        mock_product.product_id = "BTC-PERP"
        self.product_catalog.get.return_value = mock_product
        self.product_catalog.get_funding.return_value = (Decimal("0.0001"), "2024-01-01T00:00:00Z")

        result = self.service.get_product("BTC-PERP")

        assert result.funding_rate == Decimal("0.0001")
        assert result.next_funding_time == "2024-01-01T00:00:00Z"

    def test_get_product_funding_enrichment_failure_continues(self) -> None:
        """Test get_product continues if funding enrichment fails."""
        mock_product = Mock(spec=Product)
        mock_product.product_id = "BTC-PERP"
        self.product_catalog.get.return_value = mock_product
        self.product_catalog.get_funding.side_effect = KeyError("No funding")

        result = self.service.get_product("BTC-PERP")

        assert result == mock_product

    def test_get_product_catalog_fails_falls_back_to_client(self) -> None:
        """Test get_product falls back to client when catalog fails."""
        self.product_catalog.get.side_effect = KeyError("Not in catalog")
        self.client.get_product.return_value = {
            "product_id": "NEW-USD",
            "base_currency": "NEW",
            "quote_currency": "USD",
            "status": "online",
            "base_increment": "0.00000001",
            "quote_increment": "0.01",
            "base_min_size": "0.001",
        }

        result = self.service.get_product("NEW-USD")

        assert result.symbol == "NEW-USD"
        self.client.get_product.assert_called_once_with("NEW-USD")

    def test_get_product_network_error(self) -> None:
        """Test get_product handles network error."""
        self.product_catalog.get.side_effect = ConnectionError("Network error")

        result = self.service.get_product("BTC-USD")

        assert result is None
