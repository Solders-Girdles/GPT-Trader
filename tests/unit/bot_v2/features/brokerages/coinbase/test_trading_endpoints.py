"""Unit tests for CoinbaseClient trading endpoints.

Covers list_orders, list_fills, cancel_orders, and get_order_historical.
"""

import json
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient


pytestmark = pytest.mark.endpoints


def make_client() -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")


def test_list_orders_builds_query_string():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"orders": []})

    client.set_transport_for_testing(transport)
    _ = client.list_orders(product_id="BTC-USD", order_status="FILLED", start_date="2024-01-01", end_date="2024-01-31")
    url = urls[0]
    assert url.endswith("/api/v3/brokerage/orders/historical?product_id=BTC-USD&order_status=FILLED&start_date=2024-01-01&end_date=2024-01-31")


def test_list_fills_filters_by_symbol():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"fills": []})

    client.set_transport_for_testing(transport)
    _ = client.list_fills(product_id="ETH-USD", limit=100)
    assert urls[0].endswith("/api/v3/brokerage/orders/historical/fills?product_id=ETH-USD&limit=100")


def test_cancel_orders_handles_multiple_ids():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"results": [1, 2]})

    client.set_transport_for_testing(transport)
    out = client.cancel_orders(["id1", "id2"])
    method, url, payload = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/batch_cancel")
    assert payload["order_ids"] == ["id1", "id2"]
    assert out.get("results") == [1, 2]


def test_get_order_historical_formats_path():
    client = make_client()
    urls = []

    def transport(method, url, headers, body, timeout):
        urls.append(url)
        return 200, {}, json.dumps({"order": {"id": "ord-1"}})

    client.set_transport_for_testing(transport)
    _ = client.get_order_historical("ord-1")
    assert urls[0].endswith("/api/v3/brokerage/orders/historical/ord-1")


def test_list_orders_batch_handles_multiple_ids():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"orders": []})

    client.set_transport_for_testing(transport)
    out = client.list_orders_batch(["order1", "order2", "order3"])
    assert calls[0][0] == "GET"
    # Batch endpoint includes the batch path
    assert "/batch" in calls[0][1]
    assert "orders" in out


def test_place_order_sends_post():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"order_id": "new-order-123"})

    client.set_transport_for_testing(transport)
    payload = {"product_id": "BTC-USD", "side": "BUY", "order_configuration": {}}
    out = client.place_order(payload)
    
    method, url, sent_payload = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders")
    assert sent_payload == payload
    assert out["order_id"] == "new-order-123"


def test_list_orders_pagination():
    """Test that list_orders can iterate through paginated results."""
    client = make_client()
    call_count = 0
    
    def transport(method, url, headers, body, timeout):
        nonlocal call_count
        call_count += 1
        
        if call_count == 1:
            # First page with cursor
            return 200, {}, json.dumps({
                "orders": [{"id": "ord1"}, {"id": "ord2"}],
                "cursor": "next-page-cursor"
            })
        else:
            # Second page without cursor (end of pagination)
            return 200, {}, json.dumps({
                "orders": [{"id": "ord3"}],
                "cursor": None
            })
    
    client.set_transport_for_testing(transport)
    
    # Use the paginate method to iterate through all orders
    all_orders = []
    for order in client.paginate(
        path="/api/v3/brokerage/orders/historical",
        params={},
        items_key="orders"
    ):
        all_orders.append(order)
    
    assert len(all_orders) == 3
    assert all_orders[0]["id"] == "ord1"
    assert all_orders[1]["id"] == "ord2"
    assert all_orders[2]["id"] == "ord3"
    assert call_count == 2  # Two API calls made


def test_preview_order_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"preview_id": "prev-1"})

    client.set_transport_for_testing(transport)
    payload = {"product_id": "BTC-USD", "side": "BUY", "order_configuration": {"limit_limit_gtc": {"base_size": "0.1", "limit_price": "50000"}}}
    out = client.preview_order(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/preview")
    assert sent == payload
    assert out["preview_id"] == "prev-1"


def test_edit_order_preview_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"edit_preview_id": "ep-1"})

    client.set_transport_for_testing(transport)
    payload = {"order_id": "ord-1", "new_price": "49900"}
    out = client.edit_order_preview(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/edit_preview")
    assert sent == payload
    assert out["edit_preview_id"] == "ep-1"


def test_edit_order_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"success": True})

    client.set_transport_for_testing(transport)
    payload = {"order_id": "ord-1", "price": "49800"}
    out = client.edit_order(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/orders/edit")
    assert sent == payload
    assert out["success"] is True
