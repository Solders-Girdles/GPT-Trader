"""Endpoint chain tests to validate typical workflows using CoinbaseClient.

Uses mocked transport to assert correct sequencing, paths, and payloads.
"""

import json
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient

pytestmark = pytest.mark.endpoints


def make_client() -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")


def test_trading_flow_place_get_cancel():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        if url.endswith("/api/v3/brokerage/orders") and method == "POST":
            return 200, {}, json.dumps({"order_id": "ord-1"})
        if "/api/v3/brokerage/orders/historical/" in url and method == "GET":
            return 200, {}, json.dumps({"order": {"id": "ord-1", "status": "FILLED"}})
        if url.endswith("/api/v3/brokerage/orders/batch_cancel") and method == "POST":
            return 200, {}, json.dumps({"results": ["ord-1"]})
        return 200, {}, json.dumps({})

    client.set_transport_for_testing(transport)

    # Place
    place = client.place_order({"product_id": "BTC-USD", "side": "BUY", "size": "0.1"})
    # Get
    got = client.get_order_historical("ord-1")
    # Cancel
    cancelled = client.cancel_orders(["ord-1"])

    assert place["order_id"] == "ord-1"
    assert got["order"]["id"] == "ord-1"
    assert cancelled["results"] == ["ord-1"]

    # Verify path sequence and payloads
    methods_urls = [(m, u) for (m, u, _p) in calls]
    assert methods_urls[0][0] == "POST" and methods_urls[0][1].endswith("/api/v3/brokerage/orders")
    assert (
        methods_urls[1][0] == "GET"
        and "/api/v3/brokerage/orders/historical/ord-1" in methods_urls[1][1]
    )
    assert methods_urls[2][0] == "POST" and methods_urls[2][1].endswith(
        "/api/v3/brokerage/orders/batch_cancel"
    )


def test_market_data_flow_products_ticker_candles():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        if url.endswith("/api/v3/brokerage/market/products"):
            return 200, {}, json.dumps({"products": [{"product_id": "BTC-USD"}]})
        if url.endswith("/api/v3/brokerage/market/products/BTC-USD/ticker"):
            return 200, {}, json.dumps({"price": "100"})
        if url.endswith("/api/v3/brokerage/market/products/BTC-USD/candles?granularity=1H&limit=2"):
            return 200, {}, json.dumps({"candles": []})
        return 200, {}, json.dumps({})

    client.set_transport_for_testing(transport)

    prods = client.get_market_products()
    pid = prods["products"][0]["product_id"]
    _ = client.get_market_product_ticker(pid)
    _ = client.get_market_product_candles(pid, granularity="1H", limit=2)

    assert calls[0][1].endswith("/api/v3/brokerage/market/products")
    assert calls[1][1].endswith("/api/v3/brokerage/market/products/BTC-USD/ticker")
    assert calls[2][1].endswith(
        "/api/v3/brokerage/market/products/BTC-USD/candles?granularity=1H&limit=2"
    )


def test_position_flow_cfm_positions_then_detail():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        if url.endswith("/api/v3/brokerage/cfm/positions"):
            return 200, {}, json.dumps({"positions": [{"product_id": "ETH-USD-PERP"}]})
        if url.endswith("/api/v3/brokerage/cfm/positions/ETH-USD-PERP"):
            return 200, {}, json.dumps({"position": {"symbol": "ETH-USD-PERP", "size": "5"}})
        return 200, {}, json.dumps({})

    client.set_transport_for_testing(transport)

    positions = client.cfm_positions()
    sym = positions["positions"][0]["product_id"]
    detail = client.cfm_position(sym)

    assert detail["position"]["symbol"] == "ETH-USD-PERP"
    assert calls[0][1].endswith("/api/v3/brokerage/cfm/positions")
    assert calls[1][1].endswith("/api/v3/brokerage/cfm/positions/ETH-USD-PERP")
