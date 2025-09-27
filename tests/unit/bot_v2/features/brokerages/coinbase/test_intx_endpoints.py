"""Unit tests for CoinbaseClient INTX (institutional derivatives) endpoints.

Verifies HTTP method, path formatting, and mode gating for:
- intx_allocate (POST)
- intx_balances (GET)
- intx_portfolio (GET)
- intx_positions (GET)
- intx_position (GET)
- intx_multi_asset_collateral (GET)
"""

import json
import pytest

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.core.interfaces import InvalidRequestError


pytestmark = pytest.mark.endpoints


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_intx_allocate_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"allocation_id": "alloc-1"})

    client.set_transport_for_testing(transport)
    payload = {"portfolio_uuid": "port-1", "allocations": [{"symbol": "BTC-PERP", "amount": "1.0"}]}
    out = client.intx_allocate(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/intx/allocate")
    assert sent["portfolio_uuid"] == "port-1"
    assert out["allocation_id"] == "alloc-1"


def test_intx_balances_formats_path():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"balances": []})

    client.set_transport_for_testing(transport)
    _ = client.intx_balances("port-abc")
    assert calls[0].endswith("/api/v3/brokerage/intx/balances/port-abc")


def test_intx_portfolio_formats_path():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"portfolio": {}})

    client.set_transport_for_testing(transport)
    _ = client.intx_portfolio("port-xyz")
    assert calls[0].endswith("/api/v3/brokerage/intx/portfolio/port-xyz")


def test_intx_positions_formats_path():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"positions": []})

    client.set_transport_for_testing(transport)
    _ = client.intx_positions("port-xyz")
    assert calls[0].endswith("/api/v3/brokerage/intx/positions/port-xyz")


def test_intx_position_formats_path():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append(url)
        return 200, {}, json.dumps({"position": {"symbol": "ETH-PERP"}})

    client.set_transport_for_testing(transport)
    _ = client.intx_position("port-xyz", "ETH-PERP")
    assert calls[0].endswith("/api/v3/brokerage/intx/positions/port-xyz/ETH-PERP")


def test_intx_multi_asset_collateral_formats_path():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"enabled": True})

    client.set_transport_for_testing(transport)
    out = client.intx_multi_asset_collateral()
    method, url = calls[0]
    assert method == "GET"
    assert url.endswith("/api/v3/brokerage/intx/multi_asset_collateral")
    assert out["enabled"] is True


def test_intx_endpoints_blocked_in_exchange_mode():
    client = make_client(api_mode="exchange")
    with pytest.raises(InvalidRequestError):
        client.intx_multi_asset_collateral()
    with pytest.raises(InvalidRequestError):
        client.intx_positions("port-1")
