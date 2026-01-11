"""Edge coverage for Coinbase REST ProductService batch paths."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

from gpt_trader.features.brokerages.coinbase.market_data_service import MarketDataService
from gpt_trader.features.brokerages.coinbase.rest.product_service import ProductService


def _make_service() -> tuple[ProductService, Mock]:
    client = Mock()
    product_catalog = Mock()
    market_data = Mock(spec=MarketDataService)
    service = ProductService(
        client=client,
        product_catalog=product_catalog,
        market_data=market_data,
    )
    return service, client


def test_get_tickers_mid_price_uses_single_side() -> None:
    service, client = _make_service()
    client.api_mode = "advanced"
    client.get_best_bid_ask.return_value = {
        "pricebooks": [
            {
                "product_id": "BTC-USD",
                "bids": [{"price": "50000", "size": "1"}],
                "asks": [],
            },
            {
                "product_id": "ETH-USD",
                "bids": [],
                "asks": [{"price": "3000", "size": "2"}],
            },
        ]
    }

    result = service.get_tickers(["BTC-USD", "ETH-USD"])

    assert result["BTC-USD"]["price"] == "50000"
    assert result["BTC-USD"]["bid"] == "50000"
    assert result["BTC-USD"]["ask"] == "0"
    assert result["ETH-USD"]["price"] == "3000"
    assert result["ETH-USD"]["bid"] == "0"
    assert result["ETH-USD"]["ask"] == "3000"


def test_get_tickers_batch_parse_error_falls_back() -> None:
    service, client = _make_service()
    client.api_mode = "advanced"
    client.get_best_bid_ask.return_value = {
        "pricebooks": [
            {
                "product_id": "BTC-USD",
                "bids": [{"price": "bad-price", "size": "1"}],
                "asks": [{"price": "50010", "size": "1"}],
            }
        ]
    }

    def _get_ticker(product_id: str) -> dict[str, str]:
        return {"product_id": product_id, "price": "50000"}

    client.get_ticker.side_effect = _get_ticker

    result = service.get_tickers(["BTC-USD"])

    assert result["BTC-USD"]["price"] == "50000"
    client.get_best_bid_ask.assert_called_once_with(["BTC-USD"])
    client.get_ticker.assert_called_once_with("BTC-USD")


def test_get_candles_skips_invalid_time() -> None:
    service, client = _make_service()
    client.get_candles.return_value = {
        "candles": [
            {
                "time": "not-a-date",
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "volume": "10",
            },
            {
                "time": "2024-01-01T00:00:00",
                "open": "2",
                "high": "3",
                "low": "1",
                "close": "2.5",
                "volume": "20",
            },
        ]
    }

    candles = service.get_candles("BTC-USD")

    assert len(candles) == 1
    assert candles[0].ts == datetime(2024, 1, 1, 0, 0, 0)
