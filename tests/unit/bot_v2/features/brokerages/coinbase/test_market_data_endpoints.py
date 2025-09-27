"""Unit tests for CoinbaseClient market data endpoints.

Verifies endpoint paths, methods, query parameters, and basic parsing using a mocked transport.
"""

import json
from datetime import datetime

import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_get_ticker_formats_path_correctly():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url, headers, body, timeout))
        return 200, {"content-type": "application/json"}, json.dumps({"price": "123"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_ticker("BTC-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/BTC-USD/ticker")
    assert out.get("price") == "123"


def test_get_candles_includes_all_params():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"candles": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_candles(
        "ETH-USD",
        granularity="1H",
        limit=500,
        start=datetime(2024, 1, 1, 0, 0, 0),
        end=datetime(2024, 1, 2, 0, 0, 0),
    )
    url = calls[0]
    assert "granularity=1H" in url
    assert "limit=500" in url
    assert "start=2024-01-01T00:00:00Z" in url
    assert "end=2024-01-02T00:00:00Z" in url


def test_get_product_book_handles_levels_advanced():
    client = make_client(api_mode="advanced")
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"bids": [], "asks": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_product_book("BTC-USD", level=2)
    url = calls[0]
    assert url.endswith("/api/v3/brokerage/market/product_book?product_id=BTC-USD&level=2")


def test_get_product_book_handles_levels_exchange():
    client = make_client(api_mode="exchange")
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"bids": [], "asks": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_product_book("BTC-USD", level=2)
    url = calls[0]
    assert url.endswith("/products/BTC-USD/book?level=2")


def test_get_best_bid_ask_requires_advanced_mode():
    # Advanced mode works
    client = make_client(api_mode="advanced")
    calls = []
    client.set_transport_for_testing(lambda m, u, h, b, t: (200, {}, json.dumps({"data": []})))
    _ = client.get_best_bid_ask(["BTC-USD", "ETH-USD"])

    # Exchange mode raises
    client_ex = make_client(api_mode="exchange")
    try:
        client_ex.get_best_bid_ask(["BTC-USD"])  # not available in exchange
        assert False, "Expected InvalidRequestError in exchange mode"
    except Exception as e:
        assert "not available in exchange mode" in str(e)


def test_get_market_products_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"products": []})

    client.set_transport_for_testing(fake_transport)
    out = client.get_market_products()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products")
    assert "products" in out


def test_get_market_product_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"product_id": "BTC-USD"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_market_product("BTC-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/BTC-USD")
    assert "product_id" in out


def test_get_market_product_ticker_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"price": "50000"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_market_product_ticker("ETH-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/ETH-USD/ticker")
    assert "price" in out


def test_get_market_product_candles_includes_params():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"candles": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_market_product_candles("BTC-USD", granularity="5M", limit=300)
    url = calls[0]
    assert url.endswith("/api/v3/brokerage/market/products/BTC-USD/candles?granularity=5M&limit=300")


def test_get_market_product_book_formats_path():
    client = make_client(api_mode="advanced")
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"bids": [], "asks": []})

    client.set_transport_for_testing(fake_transport)
    _ = client.get_market_product_book("BTC-PERP", level=3)
    url = calls[0]
    assert url.endswith("/api/v3/brokerage/market/product_book?product_id=BTC-PERP&level=3")


def test_get_product_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"product_id": "BTC-USD", "base_currency": "BTC"})

    client.set_transport_for_testing(fake_transport)
    out = client.get_product("BTC-USD")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/market/products/BTC-USD")
    assert "product_id" in out
