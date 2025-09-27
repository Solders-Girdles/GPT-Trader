"""Unit tests for CoinbaseClient account management endpoints.

Covers get_accounts, get_account, list_portfolios, get_portfolio, and get_portfolio_breakdown.
"""

import json

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient


def make_client(api_mode: str = "advanced") -> CoinbaseClient:
    return CoinbaseClient(base_url="https://api.coinbase.com", auth=None, api_mode=api_mode)


def test_get_accounts_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"accounts": []})

    client.set_transport_for_testing(fake_transport)
    out = client.get_accounts()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/accounts")
    assert "accounts" in out


def test_get_account_includes_uuid():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"account": {"uuid": "acc-123"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_account("acc-123")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/accounts/acc-123")
    assert "account" in out


def test_list_portfolios_formats_path():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"portfolios": []})

    client.set_transport_for_testing(fake_transport)
    out = client.list_portfolios()
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/portfolios")
    assert "portfolios" in out


def test_get_portfolio_includes_uuid():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"portfolio": {"uuid": "port-456"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_portfolio("port-456")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/portfolios/port-456")
    assert "portfolio" in out


def test_get_portfolio_breakdown_includes_uuid():
    client = make_client()
    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append((method, url))
        return 200, {}, json.dumps({"breakdown": {"total_balance": "10000"}})

    client.set_transport_for_testing(fake_transport)
    out = client.get_portfolio_breakdown("port-789")
    assert calls[0][0] == "GET"
    assert calls[0][1].endswith("/api/v3/brokerage/portfolios/port-789/breakdown")
    assert "breakdown" in out


def test_move_funds_posts_payload():
    client = make_client()
    calls = []

    def transport(method, url, headers, body, timeout):
        calls.append((method, url, json.loads(body or b"{}")))
        return 200, {}, json.dumps({"transfer_id": "mv-1", "status": "completed"})

    client.set_transport_for_testing(transport)
    payload = {"from_portfolio_uuid": "port-A", "to_portfolio_uuid": "port-B", "amount": "1000", "currency": "USD"}
    out = client.move_funds(payload)
    method, url, sent = calls[0]
    assert method == "POST"
    assert url.endswith("/api/v3/brokerage/portfolios/move_funds")
    assert sent == payload
    assert out["status"] == "completed"
