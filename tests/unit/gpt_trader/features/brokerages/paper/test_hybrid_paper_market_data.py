"""Tests for HybridPaperBroker market data methods."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import MarketType
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerMarketData:
    """Test HybridPaperBroker market data methods."""

    @pytest.fixture
    def broker(self, broker_factory) -> HybridPaperBroker:
        """Create broker fixture with mocked client."""
        return broker_factory()

    def test_start_market_data_prefetches_quotes(self, broker: HybridPaperBroker) -> None:
        """Test start_market_data prefetches quotes."""
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "50000",
            "best_ask": "50100",
            "trades": [{"price": "50050"}],
        }

        broker.start_market_data(["BTC-USD", "ETH-USD"])

        assert broker._client.get_market_product_ticker.call_count == 2

    def test_stop_market_data_noop(self, broker: HybridPaperBroker) -> None:
        """Test stop_market_data is no-op."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        result = broker.stop_market_data()

        assert result is None
        assert broker._last_prices["BTC-USD"] == Decimal("50000")

    def test_get_product_from_cache(self, broker: HybridPaperBroker) -> None:
        """Test get_product returns cached product."""
        from gpt_trader.core import Product

        cached_product = Product(
            symbol="BTC-USD",
            base_asset="BTC",
            quote_asset="USD",
            market_type=MarketType.SPOT,
            min_size=Decimal("0.001"),
            step_size=Decimal("0.001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=1,
        )
        broker._products_cache["BTC-USD"] = cached_product

        result = broker.get_product("BTC-USD")

        assert result == cached_product
        broker._client.get_market_product.assert_not_called()

    def test_get_product_from_api(self, broker: HybridPaperBroker) -> None:
        """Test get_product fetches from API when not cached."""
        broker._client.get_market_product.return_value = {
            "product_id": "ETH-USD",
            "base_currency_id": "ETH",
            "quote_currency_id": "USD",
            "product_type": "SPOT",
            "base_min_size": "0.01",
            "base_increment": "0.01",
            "min_market_funds": "10",
            "quote_increment": "0.01",
        }

        result = broker.get_product("ETH-USD")

        assert result.symbol == "ETH-USD"
        assert result.base_asset == "ETH"
        assert "ETH-USD" in broker._products_cache

    def test_get_product_api_error_returns_synthetic(self, broker: HybridPaperBroker) -> None:
        """Test get_product returns synthetic product on API error."""
        broker._client.get_market_product.side_effect = Exception("API error")

        result = broker.get_product("NEW-USD")

        assert result.symbol == "NEW-USD"
        assert result.base_asset == "NEW"
        assert result.quote_asset == "USD"

    def test_get_quote_returns_quote(self, broker: HybridPaperBroker) -> None:
        """Test get_quote returns parsed quote."""
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "49900.00",
            "best_ask": "50100.00",
            "trades": [{"price": "50000.00"}],
        }

        result = broker.get_quote("BTC-USD")

        assert result.symbol == "BTC-USD"
        assert result.bid == Decimal("49900.00")
        assert result.ask == Decimal("50100.00")
        assert result.last == Decimal("50000.00")
        assert broker._last_prices["BTC-USD"] == Decimal("50000.00")

    def test_get_quote_calculates_mid_when_no_trades(self, broker: HybridPaperBroker) -> None:
        """Test get_quote calculates mid price when no trades."""
        broker._client.get_market_product_ticker.return_value = {
            "best_bid": "49900.00",
            "best_ask": "50100.00",
            "trades": [],
        }

        result = broker.get_quote("BTC-USD")

        assert result.last == Decimal("50000.00")  # (49900 + 50100) / 2

    def test_get_quote_api_error(self, broker: HybridPaperBroker) -> None:
        """Test get_quote returns None on API error."""
        broker._client.get_market_product_ticker.side_effect = Exception("API error")

        result = broker.get_quote("BTC-USD")

        assert result is None

    def test_get_ticker_returns_ticker(self, broker: HybridPaperBroker) -> None:
        """Test get_ticker returns ticker data."""
        broker._client.get_ticker.return_value = {
            "price": "50000.00",
            "volume_24h": "1000.00",
        }

        result = broker.get_ticker("BTC-USD")

        assert result["price"] == "50000.00"

    def test_get_ticker_api_error(self, broker: HybridPaperBroker) -> None:
        """Test get_ticker returns empty dict on API error."""
        broker._client.get_ticker.side_effect = Exception("API error")

        result = broker.get_ticker("BTC-USD")

        assert result == {}

    def test_get_candles_returns_candles(self, broker: HybridPaperBroker) -> None:
        """Test get_candles returns parsed candles."""
        broker._client.get_market_product_candles.return_value = {
            "candles": [
                {
                    "start": "1704067200",
                    "open": "50000",
                    "high": "51000",
                    "low": "49000",
                    "close": "50500",
                    "volume": "100",
                },
                {
                    "start": "1704070800",
                    "open": "50500",
                    "high": "52000",
                    "low": "50000",
                    "close": "51500",
                    "volume": "150",
                },
            ]
        }

        result = broker.get_candles("BTC-USD")

        assert len(result) == 2
        assert result[0].open == Decimal("50000")
        assert result[1].close == Decimal("51500")

    def test_get_candles_api_error(self, broker: HybridPaperBroker) -> None:
        """Test get_candles returns empty list on API error."""
        broker._client.get_market_product_candles.side_effect = Exception("API error")

        result = broker.get_candles("BTC-USD")

        assert result == []

    def test_list_products_returns_products(self, broker: HybridPaperBroker) -> None:
        """Test list_products returns parsed products."""
        broker._client.get_market_products.return_value = {
            "products": [
                {"product_id": "BTC-USD", "product_type": "SPOT"},
                {"product_id": "ETH-USD", "product_type": "SPOT"},
            ]
        }

        result = broker.list_products()

        assert len(result) == 2

    def test_list_products_filters_by_type(self, broker: HybridPaperBroker) -> None:
        """Test list_products filters by product type."""
        broker._client.get_market_products.return_value = {
            "products": [
                {"product_id": "BTC-USD", "product_type": "SPOT"},
                {"product_id": "BTC-PERP", "product_type": "PERPETUAL"},
            ]
        }

        result = broker.list_products(product_type="PERPETUAL")

        # Only perpetual products should be returned
        perp_products = [p for p in result if p.market_type == MarketType.PERPETUAL]
        assert len(perp_products) <= len(result)

    def test_synthetic_product_default_quote_asset(self, broker: HybridPaperBroker) -> None:
        """Test _synthetic_product uses USD as default quote asset."""
        product = broker._synthetic_product("BTC")

        assert product.base_asset == "BTC"
        assert product.quote_asset == "USD"
