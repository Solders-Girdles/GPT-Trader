"""
Comprehensive tests for ProductRestMixin.

Covers product discovery, catalog lookups, funding enrichment, and market data.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.features.brokerages.coinbase.rest.products import ProductRestMixin
from bot_v2.features.brokerages.core.interfaces import MarketType, Product, Quote


class ProductRestMixinImpl(ProductRestMixin):
    """Implementation with required attributes for testing."""

    def __init__(self):
        self.client = Mock()
        self.endpoints = Mock()
        self.product_catalog = Mock()


@pytest.fixture
def products_mixin():
    """Create products mixin instance."""
    return ProductRestMixinImpl()


@pytest.fixture
def mock_product_data():
    """Mock product data from API."""
    return {
        "product_id": "BTC-USD-PERP",
        "base_currency": "BTC",
        "quote_currency": "USD",
        "status": "online",
        "trading_disabled": False,
    }


@pytest.fixture
def mock_spot_product():
    """Mock spot product."""
    return Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )


@pytest.fixture
def mock_perp_product():
    """Mock perpetual product."""
    return Product(
        symbol="BTC-USD-PERP",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.PERPETUAL,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        price_increment=Decimal("0.01"),
        min_notional=Decimal("10"),
    )


class TestListProducts:
    """Test list_products method."""

    def test_list_products_all_markets(self, products_mixin, mock_spot_product, mock_perp_product):
        """Should list all products when no market filter."""
        products_mixin.client.get_products.return_value = {
            "products": [
                {"product_id": "BTC-USD", "status": "online"},
                {"product_id": "BTC-USD-PERP", "status": "online"},
            ]
        }
        products_mixin._enrich_with_funding = Mock(side_effect=lambda p: p)

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_product",
            side_effect=[mock_spot_product, mock_perp_product],
        ):
            products = products_mixin.list_products()

        assert len(products) == 2

    def test_list_products_filter_perpetuals(self, products_mixin, mock_perp_product):
        """Should filter for perpetual markets."""
        products_mixin.client.get_products.return_value = {
            "products": [
                {"product_id": "BTC-USD", "status": "online"},
                {"product_id": "BTC-USD-PERP", "status": "online"},
            ]
        }
        products_mixin._enrich_with_funding = Mock(side_effect=lambda p: p)

        spot = Product(
            symbol="BTC-USD",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.SPOT,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            price_increment=Decimal("0.01"),
            min_notional=Decimal("10"),
        )

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_product",
            side_effect=[spot, mock_perp_product],
        ):
            products = products_mixin.list_products(market=MarketType.PERPETUAL)

        # Should only return perpetual
        assert len(products) == 1
        assert products[0].market_type == MarketType.PERPETUAL

    def test_list_products_enriches_perpetuals(self, products_mixin, mock_perp_product):
        """Should enrich perpetual products with funding data."""
        products_mixin.client.get_products.return_value = {
            "products": [{"product_id": "BTC-USD-PERP", "status": "online"}]
        }

        enrich_mock = Mock(side_effect=lambda p: p)
        products_mixin._enrich_with_funding = enrich_mock

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_product",
            return_value=mock_perp_product,
        ):
            products_mixin.list_products()

        # Should call _enrich_with_funding for perpetuals
        enrich_mock.assert_called_once()

    def test_list_products_api_error(self, products_mixin):
        """Should handle API errors gracefully."""
        products_mixin.client.get_products.side_effect = Exception("API error")

        products = products_mixin.list_products()

        assert products == []

    def test_list_products_handles_data_key(self, products_mixin, mock_spot_product):
        """Should handle 'data' key in response."""
        products_mixin.client.get_products.return_value = {
            "data": [{"product_id": "BTC-USD", "status": "online"}]
        }
        products_mixin._enrich_with_funding = Mock(side_effect=lambda p: p)

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_product",
            return_value=mock_spot_product,
        ):
            products = products_mixin.list_products()

        assert len(products) == 1

    def test_list_products_empty_response(self, products_mixin):
        """Should handle empty response."""
        products_mixin.client.get_products.return_value = {}

        products = products_mixin.list_products()

        assert products == []


class TestGetProduct:
    """Test get_product method."""

    def test_get_product_from_catalog(self, products_mixin, mock_spot_product):
        """Should fetch product from catalog first."""
        products_mixin.product_catalog.get.return_value = mock_spot_product
        products_mixin._enrich_with_funding = Mock(side_effect=lambda p: p)

        product = products_mixin.get_product("BTC-USD")

        assert product == mock_spot_product
        products_mixin.product_catalog.get.assert_called_once()

    def test_get_product_catalog_fallback(self, products_mixin, mock_spot_product):
        """Should fallback to API if catalog lookup fails."""
        products_mixin.product_catalog.get.side_effect = Exception("Not in catalog")
        products_mixin.client.get_product.return_value = {"product_id": "BTC-USD"}
        products_mixin._enrich_with_funding = Mock(side_effect=lambda p: p)

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_product",
            return_value=mock_spot_product,
        ):
            product = products_mixin.get_product("BTC-USD")

        assert product is not None
        products_mixin.client.get_product.assert_called_once()

    def test_get_product_normalizes_symbol(self, products_mixin, mock_spot_product):
        """Should normalize symbol before lookup."""
        products_mixin.product_catalog.get.return_value = mock_spot_product
        products_mixin._enrich_with_funding = Mock(side_effect=lambda p: p)

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.normalize_symbol",
            return_value="BTC-USD",
        ) as mock_normalize:
            products_mixin.get_product("btc-usd")

        mock_normalize.assert_called_with("btc-usd")

    def test_get_product_enriches_perpetual(self, products_mixin, mock_perp_product):
        """Should enrich perpetual products."""
        products_mixin.product_catalog.get.return_value = mock_perp_product
        enrich_mock = Mock(side_effect=lambda p: p)
        products_mixin._enrich_with_funding = enrich_mock

        products_mixin.get_product("BTC-USD-PERP")

        enrich_mock.assert_called_once_with(mock_perp_product)

    def test_get_product_api_error(self, products_mixin):
        """Should return None on API error."""
        products_mixin.product_catalog.get.side_effect = Exception("Catalog error")
        products_mixin.client.get_product.side_effect = Exception("API error")

        product = products_mixin.get_product("BTC-USD")

        assert product is None

    def test_get_product_without_get_product_method(self, products_mixin, mock_spot_product):
        """Should use legacy endpoint if get_product method missing."""
        products_mixin.product_catalog.get.return_value = None
        del products_mixin.client.get_product  # Remove method

        products_mixin.client.get.return_value = {"product_id": "BTC-USD"}
        products_mixin.endpoints.get_product.return_value = "/products/BTC-USD"
        products_mixin._enrich_with_funding = Mock(side_effect=lambda p: p)

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_product",
            return_value=mock_spot_product,
        ):
            product = products_mixin.get_product("BTC-USD")

        assert product is not None


class TestEnrichWithFunding:
    """Test _enrich_with_funding method."""

    def test_enrich_with_funding_success(self, products_mixin, mock_perp_product):
        """Should enrich product with funding data."""
        products_mixin.endpoints.supports_derivatives.return_value = True
        products_mixin.product_catalog.get_funding.return_value = (
            Decimal("0.0001"),  # funding_rate
            "2024-01-15T12:00:00Z",  # next_funding_time
        )

        enriched = products_mixin._enrich_with_funding(mock_perp_product)

        assert enriched.funding_rate == Decimal("0.0001")
        assert enriched.next_funding_time == "2024-01-15T12:00:00Z"

    def test_enrich_with_funding_no_derivatives_support(self, products_mixin, mock_perp_product):
        """Should skip enrichment if derivatives not supported."""
        products_mixin.endpoints.supports_derivatives.return_value = False

        enriched = products_mixin._enrich_with_funding(mock_perp_product)

        # Should return unchanged
        assert enriched == mock_perp_product
        products_mixin.product_catalog.get_funding.assert_not_called()

    def test_enrich_with_funding_error(self, products_mixin, mock_perp_product):
        """Should handle funding fetch errors gracefully."""
        products_mixin.endpoints.supports_derivatives.return_value = True
        products_mixin.product_catalog.get_funding.side_effect = Exception("Funding error")

        enriched = products_mixin._enrich_with_funding(mock_perp_product)

        # Should return product without crashing
        assert enriched == mock_perp_product

    def test_enrich_with_funding_partial_data(self, products_mixin, mock_perp_product):
        """Should handle partial funding data."""
        products_mixin.endpoints.supports_derivatives.return_value = True
        products_mixin.product_catalog.get_funding.return_value = (
            Decimal("0.0001"),  # funding_rate
            None,  # next_funding_time missing
        )

        enriched = products_mixin._enrich_with_funding(mock_perp_product)

        # Should set available data
        assert enriched.funding_rate == Decimal("0.0001")

    def test_enrich_with_funding_none_rate(self, products_mixin, mock_perp_product):
        """Should handle None funding rate."""
        products_mixin.endpoints.supports_derivatives.return_value = True
        products_mixin.product_catalog.get_funding.return_value = (None, None)

        enriched = products_mixin._enrich_with_funding(mock_perp_product)

        # Should not crash
        assert enriched == mock_perp_product


class TestGetRestQuote:
    """Test get_rest_quote method."""

    def test_get_rest_quote_success(self, products_mixin):
        """Should fetch and convert quote."""
        products_mixin.client.get_ticker.return_value = {
            "price": "50000.50",
            "bid": "50000.00",
            "ask": "50001.00",
            "volume": "1000.5",
        }

        mock_quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("50000"),
            ask=Decimal("50001"),
            last=Decimal("50000.50"),
            ts=datetime.now(),
        )

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_quote",
            return_value=mock_quote,
        ):
            quote = products_mixin.get_rest_quote("BTC-USD")

        assert quote is not None
        assert quote.bid == Decimal("50000")

    def test_get_rest_quote_normalizes_symbol(self, products_mixin):
        """Should normalize symbol before API call."""
        products_mixin.client.get_ticker.return_value = {"price": "50000"}

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.normalize_symbol",
            return_value="BTC-USD",
        ) as mock_normalize:
            with patch("bot_v2.features.brokerages.coinbase.rest.products.to_quote"):
                products_mixin.get_rest_quote("btc-usd")

        mock_normalize.assert_called_with("btc-usd")

    def test_get_rest_quote_api_error(self, products_mixin):
        """Should return None on API error."""
        products_mixin.client.get_ticker.side_effect = Exception("API error")

        quote = products_mixin.get_rest_quote("BTC-USD")

        assert quote is None

    def test_get_rest_quote_empty_response(self, products_mixin):
        """Should handle empty response."""
        products_mixin.client.get_ticker.return_value = None

        mock_quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("0"),
            ask=Decimal("0"),
            last=Decimal("0"),
            ts=datetime.now(),
        )

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_quote",
            return_value=mock_quote,
        ):
            quote = products_mixin.get_rest_quote("BTC-USD")

        assert quote is not None


class TestGetCandles:
    """Test get_candles method."""

    def test_get_candles_success(self, products_mixin):
        """Should fetch and convert candles."""
        products_mixin.client.get_candles.return_value = {
            "candles": [
                {
                    "start": "2024-01-15T00:00:00",
                    "open": "50000",
                    "high": "50100",
                    "low": "49900",
                    "close": "50050",
                    "volume": "100",
                },
                {
                    "start": "2024-01-15T01:00:00",
                    "open": "50050",
                    "high": "50200",
                    "low": "50000",
                    "close": "50150",
                    "volume": "150",
                },
            ]
        }

        mock_candles = [Mock(), Mock()]

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_candle",
            side_effect=mock_candles,
        ):
            candles = products_mixin.get_candles("BTC-USD", "1h", limit=100)

        assert len(candles) == 2

    def test_get_candles_handles_data_key(self, products_mixin):
        """Should handle 'data' key in response."""
        products_mixin.client.get_candles.return_value = {
            "data": [
                {"start": "2024-01-15T00:00:00", "open": "50000"},
            ]
        }

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.to_candle",
            return_value=Mock(),
        ):
            candles = products_mixin.get_candles("BTC-USD", "1h")

        assert len(candles) == 1

    def test_get_candles_normalizes_symbol(self, products_mixin):
        """Should normalize symbol."""
        products_mixin.client.get_candles.return_value = {}

        with patch(
            "bot_v2.features.brokerages.coinbase.rest.products.normalize_symbol",
            return_value="BTC-USD",
        ) as mock_normalize:
            products_mixin.get_candles("btc-usd", "1h")

        mock_normalize.assert_called_with("btc-usd")

    def test_get_candles_empty_response(self, products_mixin):
        """Should handle empty response."""
        products_mixin.client.get_candles.return_value = {}

        candles = products_mixin.get_candles("BTC-USD", "1h")

        assert candles == []

    def test_get_candles_with_limit(self, products_mixin):
        """Should pass limit parameter."""
        products_mixin.client.get_candles.return_value = {}

        products_mixin.get_candles("BTC-USD", "1h", limit=50)

        products_mixin.client.get_candles.assert_called_once_with("BTC-USD", "1h", 50)


class TestGetPerpetuals:
    """Test get_perpetuals convenience method."""

    def test_get_perpetuals(self, products_mixin, mock_perp_product):
        """Should call list_products with PERPETUAL filter."""
        products_mixin.list_products = Mock(return_value=[mock_perp_product])

        perpetuals = products_mixin.get_perpetuals()

        products_mixin.list_products.assert_called_once_with(market=MarketType.PERPETUAL)
        assert len(perpetuals) == 1
        assert perpetuals[0].market_type == MarketType.PERPETUAL
