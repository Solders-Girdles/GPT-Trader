"""Pagination and query parameter edge case tests for CoinbaseClient."""

import json
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient

pytestmark = pytest.mark.endpoints


def make_client() -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")


def test_paginate_empty_pages():
    client = make_client()
    calls = []

    pages = [
        {"orders": [], "cursor": None},
    ]

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        # Return the only page (empty) every time
        return 200, {}, json.dumps(pages[0])

    client.set_transport_for_testing(transport)
    items = list(client.paginate("/api/v3/brokerage/orders/historical", params={}, items_key="orders"))
    assert items == []
    assert len(calls) == 1


def test_paginate_single_item_then_stop():
    client = make_client()
    calls = []
    sequence = [
        {"orders": [{"id": 1}], "cursor": None}
    ]
    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps(sequence[min(len(calls)-1, 0)])
    client.set_transport_for_testing(transport)
    items = list(client.paginate("/api/v3/brokerage/orders/historical", params={}, items_key="orders"))
    assert items == [{"id": 1}]
    assert len(calls) == 1


def test_query_param_encoding_edge_cases():
    client = make_client()
    urls = []
    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"ok": True})
    client.set_transport_for_testing(transport)
    # Include spaces and special characters; current client performs simple concatenation
    _ = client.list_orders(product_id="BTC USD", note="alpha+beta", tag="a/b")
    u = urls[0]
    assert "product_id=BTC USD" in u
    assert "note=alpha+beta" in u
    assert "tag=a/b" in u


def test_paginate_three_pages_with_cursor_and_next_cursor():
    client = make_client()
    calls = []
    pages = [
        {"orders": [{"id": 1}], "cursor": "A"},
        {"orders": [{"id": 2}], "next_cursor": "B"},  # variant field name
        {"orders": [{"id": 3}], "cursor": None},
    ]

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        idx = min(len(calls) - 1, len(pages) - 1)
        return 200, {}, json.dumps(pages[idx])

    client.set_transport_for_testing(transport)
    items = list(client.paginate("/api/v3/brokerage/orders/historical", params={"limit": 1}, items_key="orders"))
    assert items == [{"id": 1}, {"id": 2}, {"id": 3}]
    # Should have requested three pages
    assert len(calls) == 3
    # First call without cursor, then with cursor=A, then with cursor=B
    assert calls[0].endswith("/api/v3/brokerage/orders/historical?limit=1")
    assert "cursor=A" in calls[1]
    assert "cursor=B" in calls[2]


def test_pagination_restart_after_completion():
    client = make_client()
    calls = []
    pages = [
        {"orders": [{"id": 10}], "cursor": None},
    ]
    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps(pages[0])
    client.set_transport_for_testing(transport)
    run1 = list(client.paginate("/api/v3/brokerage/orders/historical", params={}, items_key="orders"))
    run2 = list(client.paginate("/api/v3/brokerage/orders/historical", params={}, items_key="orders"))
    assert run1 == [{"id": 10}]
    assert run2 == [{"id": 10}]
    assert len(calls) == 2
