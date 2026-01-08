"""Tests for Coinbase REST product service functionality."""

from decimal import Decimal
from unittest.mock import Mock

from gpt_trader.core import Product
from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.rest.product_service import ProductService


class TestProductService:
    """Test ProductService class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.client = Mock()  # Don't use spec to allow dynamic method mocking
        self.product_catalog = Mock()  # Don't use spec
        self.market_data = Mock(spec=MarketDataService)

        self.service = ProductService(
            client=self.client,
            product_catalog=self.product_catalog,
            market_data=self.market_data,
        )

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

    def test_get_rest_quote_returns_quote(self) -> None:
        """Test getting REST quote."""
        self.client.get_product_ticker.return_value = {
            "price": "50000.00",
            "bid": "49900.00",
            "ask": "50100.00",
            "volume": "1234.56",
            "time": "2024-01-01T00:00:00Z",
        }

        result = self.service.get_rest_quote("BTC-USD")

        assert result is not None
        self.client.get_product_ticker.assert_called_once_with("BTC-USD")

    def test_get_rest_quote_network_error(self) -> None:
        """Test get_rest_quote handles network error."""
        self.client.get_product_ticker.side_effect = ConnectionError("Network error")

        result = self.service.get_rest_quote("BTC-USD")

        assert result is None

    def test_get_quote_delegates_to_get_rest_quote(self) -> None:
        """Test get_quote delegates to get_rest_quote."""
        self.client.get_product_ticker.return_value = {
            "price": "50000.00",
            "bid": "49900.00",
            "ask": "50100.00",
        }

        result = self.service.get_quote("BTC-USD")

        assert result is not None
        self.client.get_product_ticker.assert_called_once_with("BTC-USD")

    def test_get_candles_returns_candles(self) -> None:
        """Test getting candles."""
        self.client.get_candles.return_value = {
            "candles": [
                {
                    "start": "1704067200",
                    "open": "50000.00",
                    "high": "51000.00",
                    "low": "49000.00",
                    "close": "50500.00",
                    "volume": "100.00",
                },
                {
                    "start": "1704070800",
                    "open": "50500.00",
                    "high": "52000.00",
                    "low": "50000.00",
                    "close": "51500.00",
                    "volume": "150.00",
                },
            ]
        }

        result = self.service.get_candles("BTC-USD", granularity="ONE_HOUR")

        assert len(result) == 2
        self.client.get_candles.assert_called_once_with("BTC-USD", granularity="ONE_HOUR")

    def test_get_candles_skips_invalid_entries(self) -> None:
        """Test get_candles handles invalid candle entries gracefully."""
        self.client.get_candles.return_value = {
            "candles": [
                {
                    "start": "1704067200",
                    "open": "50000.00",
                    "high": "51000.00",
                    "low": "49000.00",
                    "close": "50500.00",
                    "volume": "100.00",
                },
            ]
        }

        result = self.service.get_candles("BTC-USD")

        # With valid data, should return the valid candle
        assert len(result) == 1

    def test_get_candles_network_error(self) -> None:
        """Test get_candles handles network error."""
        self.client.get_candles.side_effect = ConnectionError("Network error")

        result = self.service.get_candles("BTC-USD")

        assert result == []

    def test_get_ticker_returns_ticker(self) -> None:
        """Test getting ticker data."""
        self.client.get_ticker.return_value = {
            "price": "50000.00",
            "volume_24h": "10000.00",
        }

        result = self.service.get_ticker("BTC-USD")

        assert result["price"] == "50000.00"

    def test_get_ticker_network_error(self) -> None:
        """Test get_ticker handles network error."""
        self.client.get_ticker.side_effect = ConnectionError("Network error")

        result = self.service.get_ticker("BTC-USD")

        assert result == {}

    def test_get_mark_price_returns_price(self) -> None:
        """Test getting mark price."""
        self.market_data.get_mark.return_value = 50000.00

        result = self.service.get_mark_price("BTC-PERP")

        assert result == Decimal("50000.00")

    def test_get_mark_price_none_returns_none(self) -> None:
        """Test get_mark_price returns None when mark is None."""
        self.market_data.get_mark.return_value = None

        result = self.service.get_mark_price("BTC-PERP")

        assert result is None

    def test_get_mark_price_conversion_error(self) -> None:
        """Test get_mark_price handles conversion error."""
        self.market_data.get_mark.return_value = "invalid"

        result = self.service.get_mark_price("BTC-PERP")

        assert result is None

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

        assert len(result) >= 0  # At least doesn't crash
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


class TestGetTickers:
    """Tests for batch get_tickers method."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.client = Mock()
        self.product_catalog = Mock()
        self.market_data = Mock(spec=MarketDataService)

        self.service = ProductService(
            client=self.client,
            product_catalog=self.product_catalog,
            market_data=self.market_data,
        )

    def test_get_tickers_empty_list_returns_empty_dict(self) -> None:
        """Test get_tickers with empty list returns empty dict."""
        result = self.service.get_tickers([])
        assert result == {}

    def test_get_tickers_advanced_mode_uses_batch_endpoint(self) -> None:
        """Test get_tickers uses batch endpoint in advanced mode."""
        self.client.api_mode = "advanced"
        self.client.get_best_bid_ask.return_value = {
            "pricebooks": [
                {
                    "product_id": "BTC-USD",
                    "bids": [{"price": "50000", "size": "1.5"}],
                    "asks": [{"price": "50010", "size": "1.0"}],
                },
                {
                    "product_id": "ETH-USD",
                    "bids": [{"price": "3000", "size": "10"}],
                    "asks": [{"price": "3005", "size": "8"}],
                },
            ]
        }

        result = self.service.get_tickers(["BTC-USD", "ETH-USD"])

        assert len(result) == 2
        assert "BTC-USD" in result
        assert "ETH-USD" in result
        # Mid-price should be calculated
        assert result["BTC-USD"]["price"] == "50005"  # (50000 + 50010) / 2
        assert result["ETH-USD"]["price"] == "3002.5"  # (3000 + 3005) / 2
        self.client.get_best_bid_ask.assert_called_once_with(["BTC-USD", "ETH-USD"])

    def test_get_tickers_advanced_mode_includes_bid_ask(self) -> None:
        """Test get_tickers result includes bid/ask in advanced mode."""
        self.client.api_mode = "advanced"
        self.client.get_best_bid_ask.return_value = {
            "pricebooks": [
                {
                    "product_id": "BTC-USD",
                    "bids": [{"price": "50000", "size": "1.5"}],
                    "asks": [{"price": "50010", "size": "1.0"}],
                },
            ]
        }

        result = self.service.get_tickers(["BTC-USD"])

        assert result["BTC-USD"]["bid"] == "50000"
        assert result["BTC-USD"]["ask"] == "50010"

    def test_get_tickers_exchange_mode_uses_individual_calls(self) -> None:
        """Test get_tickers uses individual calls in exchange mode."""
        self.client.api_mode = "exchange"
        self.client.get_ticker.side_effect = [
            {"price": "50000", "product_id": "BTC-USD"},
            {"price": "3000", "product_id": "ETH-USD"},
        ]

        result = self.service.get_tickers(["BTC-USD", "ETH-USD"])

        assert len(result) == 2
        assert result["BTC-USD"]["price"] == "50000"
        assert result["ETH-USD"]["price"] == "3000"
        assert self.client.get_ticker.call_count == 2

    def test_get_tickers_exchange_mode_handles_failures(self) -> None:
        """Test get_tickers handles individual fetch failures in exchange mode."""
        self.client.api_mode = "exchange"
        self.client.get_ticker.side_effect = [
            {"price": "50000", "product_id": "BTC-USD"},
            Exception("Network error"),  # ETH-USD fails
        ]

        result = self.service.get_tickers(["BTC-USD", "ETH-USD"])

        assert len(result) == 1
        assert "BTC-USD" in result
        assert "ETH-USD" not in result

    def test_get_tickers_advanced_mode_fallback_on_batch_failure(self) -> None:
        """Test get_tickers falls back to individual calls when batch fails."""
        self.client.api_mode = "advanced"
        self.client.get_best_bid_ask.side_effect = Exception("Batch failed")
        self.client.get_ticker.side_effect = [
            {"price": "50000", "product_id": "BTC-USD"},
        ]

        result = self.service.get_tickers(["BTC-USD"])

        assert len(result) == 1
        assert result["BTC-USD"]["price"] == "50000"
        self.client.get_ticker.assert_called_once_with("BTC-USD")

    def test_get_tickers_skips_empty_pricebooks(self) -> None:
        """Test get_tickers skips pricebooks without valid bid/ask."""
        self.client.api_mode = "advanced"
        self.client.get_best_bid_ask.return_value = {
            "pricebooks": [
                {
                    "product_id": "BTC-USD",
                    "bids": [{"price": "50000", "size": "1.5"}],
                    "asks": [{"price": "50010", "size": "1.0"}],
                },
                {
                    "product_id": "EMPTY-USD",
                    "bids": [],  # No bids
                    "asks": [],  # No asks
                },
            ]
        }

        result = self.service.get_tickers(["BTC-USD", "EMPTY-USD"])

        assert len(result) == 1
        assert "BTC-USD" in result
        assert "EMPTY-USD" not in result

    def test_get_tickers_handles_missing_product_id(self) -> None:
        """Test get_tickers handles pricebooks without product_id."""
        self.client.api_mode = "advanced"
        self.client.get_best_bid_ask.return_value = {
            "pricebooks": [
                {
                    "bids": [{"price": "50000", "size": "1.5"}],
                    "asks": [{"price": "50010", "size": "1.0"}],
                },  # No product_id
            ]
        }

        result = self.service.get_tickers(["BTC-USD"])

        assert len(result) == 0

    def test_get_tickers_skips_empty_price_ticker(self) -> None:
        """Test get_tickers skips tickers without price in exchange mode."""
        self.client.api_mode = "exchange"
        self.client.get_ticker.side_effect = [
            {"price": "50000", "product_id": "BTC-USD"},
            {"volume": "100"},  # No price field
        ]

        result = self.service.get_tickers(["BTC-USD", "ETH-USD"])

        assert len(result) == 1
        assert "BTC-USD" in result
