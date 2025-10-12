import json
import pytest
from types import SimpleNamespace

from bot_v2.features.brokerages.coinbase.client import CoinbaseClient, CoinbaseAuth
from bot_v2.features.brokerages.core.interfaces import AuthError


def make_client():
    return CoinbaseClient(
        base_url="https://api.coinbase.com", auth=CoinbaseAuth("k", "s", "p"), timeout=1
    )


# Advanced Trade product discovery should follow the documented
# `/api/v3/brokerage/products` route.
def test_request_composes_headers_and_path(monkeypatch):
    client = make_client()

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
    # Signed headers present
    h = calls[0].headers
    assert "CB-ACCESS-KEY" in h and h["CB-ACCESS-KEY"] == "k"
    # Advanced Trade mode should NOT include passphrase even if provided on auth
    assert "CB-ACCESS-PASSPHRASE" not in h
    assert "CB-ACCESS-SIGN" in h


def test_retries_on_429_then_succeeds(monkeypatch, fake_clock):
    client = make_client()

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
    client = make_client()

    def fake_transport(method, url, headers, body, timeout):
        return 401, {}, json.dumps({"code": "invalid_api_key", "message": "bad key"})

    client.set_transport_for_testing(fake_transport)
    with pytest.raises(AuthError):
        client.get_products()
