import json
from types import SimpleNamespace

import pytest

from gpt_trader.core import AuthError
from gpt_trader.features.brokerages.coinbase.client import CoinbaseAuth
from tests.unit.gpt_trader.features.brokerages.coinbase.test_helpers import make_client


# Advanced Trade product discovery should follow the documented
# `/api/v3/brokerage/products` route.
def test_request_composes_headers_and_path(monkeypatch):
    from gpt_trader.features.brokerages.coinbase.auth import SimpleAuth

    # Mock JWT generation to avoid needing a real EC key
    monkeypatch.setattr(SimpleAuth, "generate_jwt", lambda self, method, path: "mock_jwt_token")

    client = make_client(auth=CoinbaseAuth("k", "s", "p"), timeout=1)
    client.auth = SimpleAuth("test_key_name", "mock_private_key")

    calls = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(
            SimpleNamespace(method=method, url=url, headers=headers, body=body, timeout=timeout)
        )
        return 200, {"content-type": "application/json"}, json.dumps({"ok": True})

    client.set_transport_for_testing(fake_transport)
    out = client.get_products()
    assert out.get("ok") is True
    assert calls[0].method == "GET"
    # Advanced Trade 'products' endpoint path
    assert calls[0].url.endswith("/api/v3/brokerage/products")
    # JWT auth should use Bearer token in Authorization header
    h = calls[0].headers
    assert "Authorization" in h
    assert h["Authorization"] == "Bearer mock_jwt_token"


def test_retries_on_429_then_succeeds(monkeypatch, fake_clock):
    client = make_client(auth=CoinbaseAuth("k", "s", "p"), timeout=1)

    calls = {"n": 0}

    def fake_transport(method, url, headers, body, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            # First call rate limited
            return (
                429,
                {"retry-after": "0"},
                json.dumps({"error": "rate_limited", "message": "slow down"}),
            )
        return 200, {"content-type": "application/json"}, json.dumps({"ok": True})

    client.set_transport_for_testing(fake_transport)
    out = client.get_products()
    assert out.get("ok") is True
    assert calls["n"] == 2


def test_error_mapping_when_non_2xx(monkeypatch):
    client = make_client(auth=CoinbaseAuth("k", "s", "p"), timeout=1)

    def fake_transport(method, url, headers, body, timeout):
        return 401, {}, json.dumps({"code": "invalid_api_key", "message": "bad key"})

    client.set_transport_for_testing(fake_transport)
    with pytest.raises(AuthError):
        client.get_products()
