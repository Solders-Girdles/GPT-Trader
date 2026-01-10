"""Unit tests for CoinbaseClient conversion endpoints.

Covers convert_quote (POST) and references get_convert_trade (GET).
"""

import json

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient

pytestmark = pytest.mark.endpoints


def make_client() -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode="advanced")


def test_convert_quote_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"quote_id": "q-123", "expires_at": "2024-01-01T00:00:00Z"})

    client.set_transport_for_testing(transport)
    payload = {"from": "USDC", "to": "USD", "amount": "1000"}
    out = client.convert_quote(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/convert/quote")
    assert sent == payload
    assert out["quote_id"] == "q-123"


def test_get_convert_trade_already_covered_in_system_tests():
    # See tests/unit/gpt_trader/features/brokerages/coinbase/test_coinbase_integration.py::TestCoinbaseSystem
    # This placeholder ensures file cohesion without duplicating tests.
    pytest.skip("Covered by Coinbase system tests")


def test_commit_convert_trade_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"trade_id": "trade-1", "status": "pending"})

    client.set_transport_for_testing(transport)
    payload = {"amount": "100"}
    out = client.commit_convert_trade("trade-1", payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/convert/trade/trade-1")
    assert sent == payload
    assert out["trade_id"] == "trade-1"
