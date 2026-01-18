"""Tests for `ProductService.get_ticker` and `ProductService.get_mark_price`."""

from __future__ import annotations

from decimal import Decimal

from tests.unit.gpt_trader.features.brokerages.coinbase.rest.product_service_test_base import (
    ProductServiceTestBase,
)


class TestProductServiceTickerAndMarkPrice(ProductServiceTestBase):
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
