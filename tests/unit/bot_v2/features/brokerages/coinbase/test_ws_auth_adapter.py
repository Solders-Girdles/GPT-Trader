from __future__ import annotations

import os
from unittest.mock import patch

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig


class DummyAuth:
    def generate_jwt(self, method: str, path: str) -> str:
        return "test_jwt_token"


def test_adapter_ws_auth_provider_injection(monkeypatch):
    # Enable WS auth gate
    monkeypatch.setenv('COINBASE_WS_USER_AUTH', '1')

    # Build adapter
    cfg = APIConfig(
        api_key="k",
        api_secret="s",
        passphrase=None,
        base_url="https://api.coinbase.com",
        api_mode="advanced",
        sandbox=False,
        enable_derivatives=True,
        auth_type="HMAC",
    )
    adapter = CoinbaseBrokerage(cfg)

    # Inject dummy auth with generate_jwt (duck-typed)
    adapter.client.auth = DummyAuth()  # type: ignore[attr-defined]

    # Patch CoinbaseWebSocket within adapter to capture subscribe payload
    subscribe_payloads = []

    class FakeWS:
        def __init__(self, url: str, ws_auth_provider=None, **kwargs):
            self.url = url
            self._ws_auth_provider = ws_auth_provider

        def connect(self):
            pass

        def disconnect(self):
            pass

        def subscribe(self, sub):
            payload = {
                "type": "subscribe",
                "channels": sub.channels,
                "product_ids": sub.product_ids,
            }
            # Simulate ws.py subscription behavior with ws_auth_provider for user channel
            if self._ws_auth_provider and "user" in sub.channels:
                payload.update(self._ws_auth_provider())
            subscribe_payloads.append(payload)

        def stream_messages(self):
            if False:
                yield {}
            return

    with patch('bot_v2.features.brokerages.coinbase.adapter.CoinbaseWebSocket', FakeWS):
        # Trigger stream_user_events which should subscribe with JWT
        _ = list(adapter.stream_user_events())

    assert len(subscribe_payloads) == 1
    assert subscribe_payloads[0].get('jwt') == 'test_jwt_token'
