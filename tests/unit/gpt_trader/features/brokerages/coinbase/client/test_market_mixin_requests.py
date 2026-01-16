from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from gpt_trader.features.brokerages.coinbase.client.base import CoinbaseClientBase
from gpt_trader.features.brokerages.coinbase.client.market import MarketDataClientMixin
from gpt_trader.features.brokerages.coinbase.errors import InvalidRequestError


class _MarketClient(CoinbaseClientBase, MarketDataClientMixin):
    pass


def _make_market_client(api_mode: str = "advanced") -> _MarketClient:
    return _MarketClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_market_products_require_advanced_mode() -> None:
    client = _make_market_client(api_mode="exchange")

    with pytest.raises(InvalidRequestError):
        client.get_market_products()

    with pytest.raises(InvalidRequestError):
        client.get_market_product("BTC-USD")


def test_get_ticker_builds_path() -> None:
    client = _make_market_client()
    client._request = Mock(return_value={"price": "1"})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_ticker(client, "BTC-USD")

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/api/v3/brokerage/products/BTC-USD/ticker"
    )


def test_get_product_ticker_alias() -> None:
    client = _make_market_client()
    client.get_ticker = Mock(return_value={"price": "1"})  # type: ignore[attr-defined]

    result = MarketDataClientMixin.get_product_ticker(client, "BTC-USD")

    assert result == {"price": "1"}
    client.get_ticker.assert_called_once_with("BTC-USD")  # type: ignore[attr-defined]


def test_get_candles_formats_time_range() -> None:
    client = _make_market_client()
    recorded: dict[str, str] = {}

    def _request(method: str, path: str, payload=None):
        recorded["path"] = path
        return {"candles": []}

    client._request = _request  # type: ignore[method-assign]

    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=UTC)

    MarketDataClientMixin.get_candles(
        client,
        "BTC-USD",
        "ONE_MINUTE",
        limit=2,
        start=start,
        end=end,
    )

    path = recorded["path"]
    assert path.startswith("/api/v3/brokerage/products/BTC-USD/candles?")
    assert "granularity=ONE_MINUTE" in path
    assert "limit=2" in path
    assert f"start={int(start.replace(tzinfo=UTC).timestamp())}" in path
    assert f"end={int(end.timestamp())}" in path


def test_get_product_builds_path() -> None:
    client = _make_market_client()
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_product(client, "ETH-USD")

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/api/v3/brokerage/products/ETH-USD"
    )


def test_get_product_book_advanced_mode() -> None:
    client = _make_market_client(api_mode="advanced")
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_product_book(client, "BTC-USD", level=3)

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/api/v3/brokerage/product_book?product_id=BTC-USD&level=3"
    )


def test_get_product_book_exchange_mode() -> None:
    client = _make_market_client(api_mode="exchange")
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_product_book(client, "BTC-USD", level=1)

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/products/BTC-USD/book?level=1"
    )


def test_get_market_products_advanced_calls_request() -> None:
    client = _make_market_client(api_mode="advanced")
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_market_products(client)

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/api/v3/brokerage/market/products"
    )


def test_get_market_product_advanced_calls_request() -> None:
    client = _make_market_client(api_mode="advanced")
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_market_product(client, "BTC-USD")

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/api/v3/brokerage/market/products/BTC-USD"
    )


def test_get_market_product_ticker_builds_path() -> None:
    client = _make_market_client()
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_market_product_ticker(client, "BTC-USD")

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/api/v3/brokerage/market/products/BTC-USD/ticker"
    )


def test_get_market_product_candles_formats_time_range() -> None:
    client = _make_market_client()
    recorded: dict[str, str] = {}

    def _request(method: str, path: str, payload=None):
        recorded["path"] = path
        return {"candles": []}

    client._request = _request  # type: ignore[method-assign]

    start = datetime(2024, 2, 1, 0, 0, 0)
    end = datetime(2024, 2, 1, 0, 5, 0, tzinfo=UTC)

    MarketDataClientMixin.get_market_product_candles(
        client,
        "ETH-USD",
        "FIVE_MINUTE",
        start=start,
        end=end,
    )

    path = recorded["path"]
    assert path.startswith("/api/v3/brokerage/market/products/ETH-USD/candles?")
    assert "granularity=FIVE_MINUTE" in path
    assert "limit=200" in path
    assert f"start={int(start.replace(tzinfo=UTC).timestamp())}" in path
    assert f"end={int(end.timestamp())}" in path


def test_get_market_product_book_aliases_product_book() -> None:
    client = _make_market_client()
    client.get_product_book = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_market_product_book(client, "BTC-USD", level=2)

    client.get_product_book.assert_called_once_with("BTC-USD", 2)  # type: ignore[attr-defined]


def test_get_best_bid_ask_builds_query() -> None:
    client = _make_market_client(api_mode="advanced")
    client._request = Mock(return_value={})  # type: ignore[attr-defined]

    MarketDataClientMixin.get_best_bid_ask(client, ["BTC-USD", "ETH-USD"])

    client._request.assert_called_once_with(  # type: ignore[attr-defined]
        "GET", "/api/v3/brokerage/best_bid_ask?product_ids=BTC-USD,ETH-USD"
    )


def test_get_best_bid_ask_exchange_mode_raises() -> None:
    client = _make_market_client(api_mode="exchange")

    with pytest.raises(InvalidRequestError):
        MarketDataClientMixin.get_best_bid_ask(client, ["BTC-USD"])
