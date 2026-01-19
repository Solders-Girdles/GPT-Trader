"""Tests for `ProductService.get_tickers`."""

from __future__ import annotations

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.product_service_test_base import (
    ProductServiceTestBase,
)


class TestGetTickers(ProductServiceTestBase):
    """Tests for batch get_tickers method."""

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
