from __future__ import annotations

import json

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient


class _DummyAuth:
    def __init__(self) -> None:
        self.called = False

    def get_headers(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.called = True
        return {"Authorization": "Bearer test"}


def test_get_ticker_uses_public_endpoint_when_unauthenticated() -> None:
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        return 200, {}, json.dumps({"price": "100", "bid": "99", "ask": "101"})

    client.set_transport_for_testing(transport)

    _ = client.get_ticker("BTC-USD")

    assert len(calls) == 1
    assert calls[0].endswith("/api/v3/brokerage/market/products/BTC-USD/ticker")


def test_get_candles_uses_public_endpoint_when_unauthenticated() -> None:
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        return 200, {}, json.dumps({"candles": []})

    client.set_transport_for_testing(transport)

    _ = client.get_candles("BTC-USD", "1H", limit=2)

    assert len(calls) == 1
    assert calls[0].endswith(
        "/api/v3/brokerage/market/products/BTC-USD/candles?granularity=1H&limit=2"
    )


def test_get_ticker_prefers_public_endpoint_even_when_authenticated() -> None:
    auth = _DummyAuth()
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        return 200, {}, json.dumps({"price": "100", "bid": "99", "ask": "101"})

    client.set_transport_for_testing(transport)

    result = client.get_ticker("BTC-USD")

    assert result.get("price") == "100"
    assert len(calls) == 1
    assert calls[0].endswith("/api/v3/brokerage/market/products/BTC-USD/ticker")
    assert auth.called is False


def test_get_ticker_falls_back_to_authenticated_when_public_not_found() -> None:
    auth = _DummyAuth()
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        if url.endswith("/api/v3/brokerage/market/products/BTC-USD/ticker"):
            return 404, {}, json.dumps({"message": "not found"})
        return 200, {}, json.dumps({"price": "123", "bid": "122", "ask": "124"})

    client.set_transport_for_testing(transport)

    result = client.get_ticker("BTC-USD")

    assert result.get("price") == "123"
    assert len(calls) == 2
    assert calls[0].endswith("/api/v3/brokerage/market/products/BTC-USD/ticker")
    assert calls[1].endswith("/api/v3/brokerage/products/BTC-USD/ticker")
    assert auth.called is True


def test_get_candles_prefers_public_endpoint_even_when_authenticated() -> None:
    auth = _DummyAuth()
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        return 200, {}, json.dumps({"candles": []})

    client.set_transport_for_testing(transport)

    result = client.get_candles("BTC-USD", "1H", limit=2)

    assert result.get("candles") == []
    assert len(calls) == 1
    assert calls[0].endswith(
        "/api/v3/brokerage/market/products/BTC-USD/candles?granularity=1H&limit=2"
    )
    assert auth.called is False


def test_get_candles_falls_back_to_authenticated_when_public_not_found() -> None:
    auth = _DummyAuth()
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")
    calls: list[str] = []

    def transport(method, url, headers, body, timeout):  # noqa: ANN001, ANN002, ANN003
        calls.append(url)
        if url.endswith("/api/v3/brokerage/market/products/BTC-USD/candles?granularity=1H&limit=2"):
            return 404, {}, json.dumps({"message": "not found"})
        return 200, {}, json.dumps({"candles": []})

    client.set_transport_for_testing(transport)

    result = client.get_candles("BTC-USD", "1H", limit=2)

    assert result.get("candles") == []
    assert len(calls) == 2
    assert calls[0].endswith(
        "/api/v3/brokerage/market/products/BTC-USD/candles?granularity=1H&limit=2"
    )
    assert calls[1].endswith("/api/v3/brokerage/products/BTC-USD/candles?granularity=1H&limit=2")
    assert auth.called is True
