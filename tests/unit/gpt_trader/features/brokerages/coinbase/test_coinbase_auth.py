"""Consolidated Coinbase authentication tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient
from gpt_trader.features.brokerages.coinbase.auth import CDPJWTAuth, HMACAuth
from gpt_trader.features.brokerages.coinbase.client import CoinbaseAuth, CoinbaseClient
from gpt_trader.features.brokerages.coinbase.models import APIConfig
from tests.unit.gpt_trader.features.brokerages.coinbase.test_helpers import CoinbaseBrokerage


def test_sign_headers_deterministic(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_700_000_000)
    # Use valid base64 secret
    auth = HMACAuth(api_key="k", api_secret="c2VjcmV0", passphrase="p")
    headers = auth.sign("POST", "/api/v3/brokerage/orders", {"a": 1})
    assert headers["CB-ACCESS-KEY"] == "k"
    assert headers["CB-ACCESS-PASSPHRASE"] == "p"
    assert headers["CB-ACCESS-TIMESTAMP"] == str(1_700_000_000)
    assert headers["CB-ACCESS-SIGN"]
    assert headers["Content-Type"] == "application/json"


def test_auth_inherits_client_api_mode_and_json_body(monkeypatch):
    auth = HMACAuth(api_key="k", api_secret="c2VjcmV0", passphrase="p")
    client = CoinbaseClient(base_url="https://api.coinbase.com", auth=auth, api_mode="advanced")

    calls: list[SimpleNamespace] = []

    def fake_transport(method, url, headers, body, timeout):
        calls.append(
            SimpleNamespace(method=method, url=url, headers=headers, body=body, timeout=timeout)
        )
        return 200, {"content-type": "application/json"}, json.dumps({"ok": True})

    client.set_transport_for_testing(fake_transport)
    client._request("POST", "/api/v3/brokerage/orders", payload=None)  # type: ignore[arg-type]

    c = calls[0]
    assert "CB-ACCESS-KEY" in c.headers
    assert "CB-ACCESS-PASSPHRASE" in c.headers  # advanced mode with HMAC keeps passphrase
    if c.body:
        assert c.body.decode("utf-8") == "{}"
    else:
        assert c.body is None


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
            HMACAuth,
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


    # assert subscribe_payloads[0].get("jwt") == "test_jwt_token"
    pass
