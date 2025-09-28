"""Consolidated Coinbase authentication tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.auth import CDPJWTAuth
from bot_v2.features.brokerages.coinbase.client import CoinbaseAuth, CoinbaseClient
from bot_v2.features.brokerages.coinbase.models import APIConfig


def test_sign_headers_deterministic(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_700_000_000)
    auth = CoinbaseAuth(api_key="k", api_secret="s", passphrase="p")
    headers = auth.sign("POST", "/api/v3/brokerage/orders", {"a": 1})
    assert headers["CB-ACCESS-KEY"] == "k"
    assert headers["CB-ACCESS-PASSPHRASE"] == "p"
    assert headers["CB-ACCESS-TIMESTAMP"] == str(1_700_000_000)
    assert headers["CB-ACCESS-SIGN"]
    assert headers["Content-Type"] == "application/json"


def test_auth_inherits_client_api_mode_and_json_body(monkeypatch):
    auth = CoinbaseAuth(api_key="k", api_secret="s", passphrase="p")
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")

    calls: list[SimpleNamespace] = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(
            SimpleNamespace(method=method, url=url, headers=headers, body=body, timeout=timeout)
        )
        return 200, {"content-type": "application/json"}, json.dumps({"ok": True})

    client.set_transport_for_testing(fake_transport)
    client._request("POST", "/api/v3/brokerage/orders", body=None)  # type: ignore[arg-type]

    c = calls[0]
    assert "CB-ACCESS-KEY" in c.headers
    assert "CB-ACCESS-PASSPHRASE" not in c.headers  # advanced mode drops passphrase
    assert c.body.decode("utf-8") == "{}"


@pytest.mark.parametrize(
    "config_kwargs,expected_auth_type",
    [
        (
            {
                "api_key": "",
                "api_secret": "",
                "passphrase": None,
                "cdp_api_key": "key",
                "cdp_private_key": "secret",
            },
            CDPJWTAuth,
        ),
        (
            {"api_key": "key", "api_secret": "secret", "passphrase": "p"},
            CoinbaseAuth,
        ),
    ],
)
def test_broker_auth_selection(config_kwargs, expected_auth_type):
    config = APIConfig(
        base_url="https://api.coinbase.com",
        sandbox=True,
        api_mode="advanced",
        enable_derivatives=True,
        **config_kwargs,
    )
    broker = CoinbaseBrokerage(config)
    assert isinstance(broker.client.auth, expected_auth_type)


def test_adapter_ws_auth_provider_injection(monkeypatch):
    monkeypatch.setenv("COINBASE_WS_USER_AUTH", "1")

    config = APIConfig(
        api_key="k",
        api_secret="s",
        passphrase=None,
        base_url="https://api.coinbase.com",
        api_mode="advanced",
        sandbox=False,
        enable_derivatives=True,
        auth_type="HMAC",
    )
    adapter = CoinbaseBrokerage(config)

    class DummyAuth:
        def generate_jwt(self, method: str, path: str) -> str:
            return "test_jwt_token"

    adapter.client.auth = DummyAuth()  # type: ignore[attr-defined]

    subscribe_payloads: list[dict] = []

    class FakeWS:
        def __init__(self, url: str, ws_auth_provider=None, **kwargs):
            self._ws_auth_provider = ws_auth_provider

        def connect(self):
            pass

        def disconnect(self):
            pass

        def subscribe(self, subscription):
            payload = {
                "type": "subscribe",
                "channels": subscription.channels,
                "product_ids": subscription.product_ids,
            }
            if self._ws_auth_provider and "user" in subscription.channels:
                payload.update(self._ws_auth_provider())
            subscribe_payloads.append(payload)

        def stream_messages(self):
            if False:
                yield {}

    with patch("bot_v2.features.brokerages.coinbase.adapter.CoinbaseWebSocket", FakeWS):
        list(adapter.stream_user_events())

    assert subscribe_payloads[0].get("jwt") == "test_jwt_token"
