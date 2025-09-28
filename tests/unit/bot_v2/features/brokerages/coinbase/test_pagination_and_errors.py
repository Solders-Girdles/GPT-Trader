import json

import pytest

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

pytestmark = pytest.mark.endpoints


def make_client() -> CoinbaseClient:
    return CoinbaseClient(
        base_url="https://api.coinbase.com",
        auth=None,
        api_mode="advanced",
    )


def test_paginate_yields_all_items(monkeypatch):
    client = CoinbaseClient("https://api", CoinbaseAuth("k", "s"))

    pages = [
        (200, {}, json.dumps({"orders": [{"id": "o1"}, {"id": "o2"}], "cursor": "abc"})),
        (200, {}, json.dumps({"orders": [{"id": "o3"}]})),
    ]
    idx = {"i": 0}

    def transport(method, url, headers, body, timeout):
        i = idx["i"]
        idx["i"] += 1
        return pages[i]

    client.set_transport_for_testing(transport)
    items = list(
        client.paginate(
            "/api/v3/brokerage/orders/historical",
            {"limit": 2},
            items_key="orders",
        )
    )
    assert [item["id"] for item in items] == ["o1", "o2", "o3"]


def test_paginate_empty_pages():
    client = make_client()
    calls = []

    pages = [
        {"orders": [], "cursor": None},
    ]

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps(pages[0])

    client.set_transport_for_testing(transport)
    items = list(
        client.paginate(
            "/api/v3/brokerage/orders/historical",
            params={},
            items_key="orders",
        )
    )
    assert items == []
    assert len(calls) == 1


def test_paginate_single_item_then_stop():
    client = make_client()
    calls = []
    sequence = [{"orders": [{"id": 1}], "cursor": None}]

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps(sequence[min(len(calls) - 1, 0)])

    client.set_transport_for_testing(transport)
    items = list(
        client.paginate(
            "/api/v3/brokerage/orders/historical",
            params={},
            items_key="orders",
        )
    )
    assert items == [{"id": 1}]
    assert len(calls) == 1


def test_query_param_encoding_edge_cases():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"ok": True})

    client.set_transport_for_testing(transport)
    _ = client.list_orders(product_id="BTC USD", note="alpha+beta", tag="a/b")
    url = urls[0]
    assert "product_id=BTC USD" in url
    assert "note=alpha+beta" in url
    assert "tag=a/b" in url


def test_paginate_three_pages_with_cursor_and_next_cursor():
    client = make_client()
    calls = []
    pages = [
        {"orders": [{"id": 1}], "cursor": "A"},
        {"orders": [{"id": 2}], "next_cursor": "B"},
        {"orders": [{"id": 3}], "cursor": None},
    ]

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        idx = min(len(calls) - 1, len(pages) - 1)
        return 200, {}, json.dumps(pages[idx])

    client.set_transport_for_testing(transport)
    items = list(
        client.paginate(
            "/api/v3/brokerage/orders/historical",
            params={"limit": 1},
            items_key="orders",
        )
    )
    assert items == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert len(calls) == 3
    assert calls[0].endswith("/api/v3/brokerage/orders/historical?limit=1")
    assert "cursor=A" in calls[1]
    assert "cursor=B" in calls[2]


def test_pagination_restart_after_completion():
    client = make_client()
    calls = []
    page = {"orders": [{"id": 10}], "cursor": None}

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps(page)

    client.set_transport_for_testing(transport)
    run1 = list(
        client.paginate(
            "/api/v3/brokerage/orders/historical",
            params={},
            items_key="orders",
        )
    )
    run2 = list(
        client.paginate(
            "/api/v3/brokerage/orders/historical",
            params={},
            items_key="orders",
        )
    )
    assert run1 == [{"id": 10}]
    assert run2 == [{"id": 10}]
    assert len(calls) == 2


def test_error_mapping_variants():
    assert isinstance(map_http_error(429, None, "rate limit exceeded"), RateLimitError)
    assert isinstance(map_http_error(401, None, "bad key"), AuthError)
    assert isinstance(map_http_error(404, None, "nope"), NotFoundError)
    assert isinstance(map_http_error(400, None, "invalid size"), InvalidRequestError)
    assert isinstance(map_http_error(403, None, "forbidden"), PermissionDeniedError)
    assert isinstance(map_http_error(400, "insufficient_funds", None), InsufficientFunds)
