import json

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth
from bot_v2.features.brokerages.coinbase.errors import map_http_error
from bot_v2.features.brokerages.core.interfaces import (
    RateLimitError,
    AuthError,
    NotFoundError,
    InvalidRequestError,
    InsufficientFunds,
    PermissionDeniedError,
)


def test_paginate_yields_all_items(monkeypatch):
    c = CoinbaseClient("https://api", CoinbaseAuth("k", "s"))

    pages = [
        (200, {}, json.dumps({"orders": [{"id": "o1"}, {"id": "o2"}], "cursor": "abc"})),
        (200, {}, json.dumps({"orders": [{"id": "o3"}]})),
    ]
    idx = {"i": 0}

    def tx(method, url, headers, body, timeout):
        i = idx["i"]
        idx["i"] += 1
        return pages[i]

    c.set_transport_for_testing(tx)
    items = list(c.paginate("/api/v3/brokerage/orders/historical", {"limit": 2}, items_key="orders"))
    assert [it["id"] for it in items] == ["o1", "o2", "o3"]


def test_error_mapping_variants():
    assert isinstance(map_http_error(429, None, "rate limit exceeded"), RateLimitError)
    assert isinstance(map_http_error(401, None, "bad key"), AuthError)
    assert isinstance(map_http_error(404, None, "nope"), NotFoundError)
    assert isinstance(map_http_error(400, None, "invalid size"), InvalidRequestError)
    assert isinstance(map_http_error(403, None, "forbidden"), PermissionDeniedError)
    assert isinstance(map_http_error(400, "insufficient_funds", None), InsufficientFunds)

