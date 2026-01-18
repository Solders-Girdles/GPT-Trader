"""Tests for `ProductService` quote and candle methods."""

from __future__ import annotations

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.product_service_test_base import (
    ProductServiceTestBase,
)


class TestProductServiceQuotesAndCandles(ProductServiceTestBase):
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
