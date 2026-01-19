"""Tests for `MarketDataService` alias in `coinbase/market_data_service.py`."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.market_data_service import (
    CoinbaseTickerService,
    MarketDataService,
)


class TestMarketDataServiceAlias:
    """Tests for MarketDataService alias."""

    def test_alias_is_coinbase_ticker_service(self) -> None:
        """Test that MarketDataService is an alias for CoinbaseTickerService."""
        assert MarketDataService is CoinbaseTickerService

    def test_alias_instantiation(self) -> None:
        """Test that MarketDataService can be instantiated."""
        service = MarketDataService(symbols=["BTC-USD"])

        assert isinstance(service, CoinbaseTickerService)
        assert service._symbols == ["BTC-USD"]
