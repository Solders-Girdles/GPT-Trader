"""WebSocket subscription tests for Coinbase WS client.

Verifies payload construction for channels, product_ids, and auth via ws_auth_provider.
"""

import pytest

from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket, WSSubscription

pytestmark = pytest.mark.endpoints


class CaptureTransport:
    def __init__(self):
        self.connected = False
        self.subscriptions = []

    def connect(self, url, headers=None):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def subscribe(self, payload):
        self.subscriptions.append(payload)

    def stream(self):
        if False:
            yield {}


def test_ws_subscribe_includes_channels_and_products():
    t = CaptureTransport()
    ws = CoinbaseWebSocket(url="wss://test", transport=t)
    ws.connect()

    sub = WSSubscription(channels=["ticker"], product_ids=["BTC-USD"])
    ws.subscribe(sub)

    assert len(t.subscriptions) == 1
    p = t.subscriptions[0]
    assert p["type"] == "subscribe"
    assert p["channels"] == ["ticker"]
    assert p["product_ids"] == ["BTC-USD"]


def test_ws_subscribe_includes_auth_data_on_user_channel():
    t = CaptureTransport()
    ws = CoinbaseWebSocket(url="wss://test", transport=t, ws_auth_provider=lambda: {"jwt": "token"})
    ws.connect()

    sub = WSSubscription(channels=["user"], product_ids=[])
    ws.subscribe(sub)

    p = t.subscriptions[0]
    assert p["channels"] == ["user"]
    assert p["jwt"] == "token"


def test_ws_subscribe_prefers_explicit_auth_data():
    t = CaptureTransport()
    ws = CoinbaseWebSocket(url="wss://test", transport=t, ws_auth_provider=lambda: {"jwt": "prov"})
    ws.connect()

    sub = WSSubscription(channels=["user"], product_ids=[], auth_data={"jwt": "explicit"})
    ws.subscribe(sub)

    p = t.subscriptions[0]
    assert p["jwt"] == "explicit"

