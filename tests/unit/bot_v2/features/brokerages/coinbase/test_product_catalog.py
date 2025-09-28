"""
Unit tests for Phase 2 - Product Catalog & Metadata.
Tests product enrichment, catalog operations, and rule enforcement without network calls.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from bot_v2.features.brokerages.coinbase.utilities import (
    ProductCatalog,
    enforce_perp_rules,
    quantize_to_increment,
)
from bot_v2.features.brokerages.coinbase.models import to_product
from bot_v2.features.brokerages.core.interfaces import Product, MarketType
from bot_v2.features.brokerages.coinbase.errors import NotFoundError, InvalidRequestError


class TestProductMapping:
    """Test that to_product correctly maps perps-specific fields."""

    def test_to_product_spot_market(self):
        """Test spot product mapping (no perps fields)."""
        payload = {
            "product_id": "BTC-USD",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "min_notional": "10",
        }

        product = to_product(payload)

        assert product.symbol == "BTC-USD"
        assert product.market_type == MarketType.SPOT
        assert product.contract_size is None
        assert product.funding_rate is None
        assert product.next_funding_time is None
        assert product.leverage_max is None

    def test_to_product_perpetual_full(self):
        """Test perpetual product mapping with all fields."""
        payload = {
            "product_id": "BTC-PERP",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "min_notional": "10",
            "max_leverage": 20,
            "contract_size": "1",
            "funding_rate": "0.0001",
            "next_funding_time": "2024-01-15T16:00:00Z",
        }

        product = to_product(payload)

        assert product.symbol == "BTC-PERP"
        assert product.market_type == MarketType.PERPETUAL
        assert product.contract_size == Decimal("1")
        assert product.funding_rate == Decimal("0.0001")
        assert product.next_funding_time == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)
        assert product.leverage_max == 20

    def test_to_product_perpetual_partial(self):
        """Test perpetual with missing optional fields."""
        payload = {
            "product_id": "ETH-PERP",
            "base_currency": "ETH",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.01",
            "base_increment": "0.001",
            "quote_increment": "0.1",
        }

        product = to_product(payload)

        assert product.symbol == "ETH-PERP"
        assert product.market_type == MarketType.PERPETUAL
        assert product.contract_size is None  # Uses default None
        assert product.funding_rate is None  # Uses default None
        assert product.next_funding_time is None
        assert product.leverage_max is None

    def test_to_product_future_market(self):
        """Test futures product mapping."""
        payload = {
            "product_id": "BTC-USD-240331",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "future",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "expiry": "2024-03-31T08:00:00Z",
            "contract_size": "1",
        }

        product = to_product(payload)

        assert product.symbol == "BTC-USD-240331"
        assert product.market_type == MarketType.FUTURES
        assert product.expiry == datetime(2024, 3, 31, 8, 0, 0, tzinfo=timezone.utc)
        assert product.contract_size == Decimal("1")

    def test_to_product_invalid_funding_time(self):
        """Test handling of invalid funding time format."""
        payload = {
            "product_id": "BTC-PERP",
            "base_currency": "BTC",
            "quote_currency": "USD",
            "contract_type": "perpetual",
            "base_min_size": "0.001",
            "base_increment": "0.00001",
            "quote_increment": "0.01",
            "next_funding_time": "invalid-date",
        }

        product = to_product(payload)

        # Should handle gracefully
        assert product.next_funding_time is None


class TestProductCatalog:
    """Test ProductCatalog with perps metadata."""

    def make_catalog(self, ttl_seconds: int = 900) -> ProductCatalog:
        """Create a catalog instance."""
        return ProductCatalog(ttl_seconds=ttl_seconds)

    def test_catalog_refresh_with_perps(self):
        """Test refresh caches perpetual products."""
        catalog = self.make_catalog()

        # Mock client
        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                },
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                    "contract_size": "1",
                    "funding_rate": "0.0001",
                    "next_funding_time": "2024-01-15T16:00:00Z",
                    "max_leverage": 20,
                },
            ]
        }

        catalog.refresh(mock_client)

        # Verify both products cached
        assert len(catalog._cache) == 2
        assert "BTC-USD" in catalog._cache
        assert "BTC-PERP" in catalog._cache

        # Verify perp has metadata
        perp = catalog._cache["BTC-PERP"]
        assert perp.market_type == MarketType.PERPETUAL
        assert perp.contract_size == Decimal("1")
        assert perp.funding_rate == Decimal("0.0001")
        assert perp.leverage_max == 20

    def test_catalog_get_with_expiry(self):
        """Test get refreshes when expired."""
        catalog = self.make_catalog(ttl_seconds=1)

        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                }
            ]
        }

        # First get - should refresh
        product = catalog.get(mock_client, "BTC-PERP")
        assert product.symbol == "BTC-PERP"
        assert mock_client.get_products.call_count == 1

        # Immediate second get - should use cache
        product = catalog.get(mock_client, "BTC-PERP")
        assert mock_client.get_products.call_count == 1

        # Simulate expiry without sleeping by moving last_refresh back in time
        from datetime import datetime, timedelta

        catalog._last_refresh = datetime.utcnow() - timedelta(seconds=2)

        # Third get - should refresh again
        product = catalog.get(mock_client, "BTC-PERP")
        assert mock_client.get_products.call_count == 2

    def test_catalog_get_not_found(self):
        """Test get raises NotFoundError for missing product."""
        catalog = self.make_catalog()

        mock_client = MagicMock()
        mock_client.get_products.return_value = {"products": []}

        with pytest.raises(NotFoundError) as exc_info:
            catalog.get(mock_client, "MISSING-PERP")

        assert "Product not found: MISSING-PERP" in str(exc_info.value)
        # Should have tried refresh twice
        assert mock_client.get_products.call_count == 2

    def test_catalog_get_funding_for_perpetual(self):
        """Test get_funding returns data for perpetuals."""
        catalog = self.make_catalog()

        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-PERP",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                    "funding_rate": "0.0001",
                    "next_funding_time": "2024-01-15T16:00:00Z",
                }
            ]
        }

        funding_rate, next_funding = catalog.get_funding(mock_client, "BTC-PERP")

        assert funding_rate == Decimal("0.0001")
        assert next_funding == datetime(2024, 1, 15, 16, 0, 0, tzinfo=timezone.utc)

    def test_catalog_get_funding_for_spot(self):
        """Test get_funding returns None for spot products."""
        catalog = self.make_catalog()

        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "products": [
                {
                    "product_id": "BTC-USD",
                    "base_currency": "BTC",
                    "quote_currency": "USD",
                    "base_min_size": "0.001",
                    "base_increment": "0.00001",
                    "quote_increment": "0.01",
                }
            ]
        }

        funding_rate, next_funding = catalog.get_funding(mock_client, "BTC-USD")

        assert funding_rate is None
        assert next_funding is None

    def test_catalog_handles_alternative_response_format(self):
        """Test catalog handles 'data' key instead of 'products'."""
        catalog = self.make_catalog()

        mock_client = MagicMock()
        mock_client.get_products.return_value = {
            "data": [  # Alternative format
                {
                    "product_id": "ETH-PERP",
                    "base_currency": "ETH",
                    "quote_currency": "USD",
                    "contract_type": "perpetual",
                    "base_min_size": "0.01",
                    "base_increment": "0.001",
                    "quote_increment": "0.1",
                }
            ]
        }

        catalog.refresh(mock_client)

        assert "ETH-PERP" in catalog._cache
        assert catalog._cache["ETH-PERP"].market_type == MarketType.PERPETUAL


class TestEnforcePerpRules:
    """Test enforce_perp_rules helper."""

    def make_perp_product(self) -> Product:
        """Create a test perpetual product."""
        return Product(
            symbol="BTC-PERP",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.00001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=20,
            contract_size=Decimal("1"),
        )

    def test_enforce_quantizes_qty(self):
        """Test quantity is quantized to step_size."""
        product = self.make_perp_product()

        # Input not aligned to step_size
        qty, price = enforce_perp_rules(product, Decimal("0.123456789"))

        # Should be quantized down to nearest step
        assert qty == Decimal("0.12345")  # 0.123456789 -> 0.12345 (step 0.00001)
        assert price is None

    def test_enforce_quantizes_price(self):
        """Test price is quantized to price_increment."""
        product = self.make_perp_product()

        # Input price not aligned
        qty, price = enforce_perp_rules(product, Decimal("0.01"), Decimal("50123.456"))

        assert qty == Decimal("0.01")
        assert price == Decimal("50123.45")  # Quantized to 0.01 increment

    def test_enforce_rejects_below_min_size(self):
        """Test rejection when qty below min_size."""
        product = self.make_perp_product()

        with pytest.raises(InvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.0001"))  # Below 0.001

        assert "below minimum size" in str(exc_info.value)
        assert "0.001" in str(exc_info.value)

    def test_enforce_rejects_below_min_notional(self):
        """Test rejection when notional below minimum."""
        product = self.make_perp_product()

        # qty * price = 0.001 * 100 = 0.1 (below min_notional of 10)
        with pytest.raises(InvalidRequestError) as exc_info:
            enforce_perp_rules(product, Decimal("0.001"), Decimal("100"))

        assert "below minimum" in str(exc_info.value)
        assert "10" in str(exc_info.value)

    def test_enforce_accepts_valid_notional(self):
        """Test acceptance when notional meets minimum."""
        product = self.make_perp_product()

        # qty * price = 0.001 * 20000 = 20 (above min_notional of 10)
        qty, price = enforce_perp_rules(product, Decimal("0.001"), Decimal("20000"))

        assert qty == Decimal("0.001")
        assert price == Decimal("20000")

    def test_enforce_handles_no_min_notional(self):
        """Test handling when product has no min_notional."""
        product = self.make_perp_product()
        product.min_notional = None

        # Should not check notional
        qty, price = enforce_perp_rules(
            product, Decimal("0.001"), Decimal("1")  # Very low notional, but should pass
        )

        assert qty == Decimal("0.001")
        assert price == Decimal("1")

    def test_enforce_complex_quantization(self):
        """Test complex quantization scenario."""
        product = Product(
            symbol="ETH-PERP",
            base_asset="ETH",
            quote_asset="USD",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.01"),
            step_size=Decimal("0.001"),  # 0.001 ETH increments
            min_notional=Decimal("50"),
            price_increment=Decimal("0.1"),  # $0.10 increments
            leverage_max=15,
        )

        # Input: 0.123456 ETH at $2345.67
        qty, price = enforce_perp_rules(product, Decimal("0.123456"), Decimal("2345.67"))

        # Expected: 0.123 ETH (floored to 0.001) at $2345.6 (floored to 0.1)
        assert qty == Decimal("0.123")
        assert price == Decimal("2345.6")

        # Verify notional passes: 0.123 * 2345.6 = 288.5 > 50
        notional = qty * price
        assert notional >= product.min_notional


class TestQuantizeToIncrement:
    """Test the quantize_to_increment helper."""

    def test_quantize_basic(self):
        """Test basic quantization."""
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0.01"))
        assert result == Decimal("1.23")

    def test_quantize_floors_not_rounds(self):
        """Test quantization floors, doesn't round."""
        result = quantize_to_increment(Decimal("1.2389"), Decimal("0.01"))
        assert result == Decimal("1.23")  # Not 1.24

    def test_quantize_handles_zero_increment(self):
        """Test handling of zero/None increment."""
        result = quantize_to_increment(Decimal("1.2345"), Decimal("0"))
        assert result == Decimal("1.2345")

        result = quantize_to_increment(Decimal("1.2345"), None)
        assert result == Decimal("1.2345")

    def test_quantize_arbitrary_increments(self):
        """Test non-power-of-10 increments."""
        # Increment of 0.025
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.025"))
        assert result == Decimal("1.225")  # 49 * 0.025

        # Increment of 0.005
        result = quantize_to_increment(Decimal("1.237"), Decimal("0.005"))
        assert result == Decimal("1.235")  # 247 * 0.005
