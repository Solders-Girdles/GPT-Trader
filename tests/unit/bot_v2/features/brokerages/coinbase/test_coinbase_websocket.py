"""Consolidated Coinbase WebSocket tests covering subscriptions, streaming, and reconnects."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Any, Deque, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.transports import MockTransport
from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    SequenceGuard,
    WSSubscription,
    normalize_market_message,
)

pytestmark = pytest.mark.endpoints


def make_adapter(**overrides: Any) -> CoinbaseBrokerage:
    config = APIConfig(
        api_key=overrides.get("api_key", "test"),
        api_secret=overrides.get("api_secret", "test"),
        passphrase=overrides.get("passphrase"),
        base_url=overrides.get("base_url", "https://api.coinbase.com"),
        sandbox=overrides.get("sandbox", False),
        api_mode=overrides.get("api_mode", "advanced"),
        enable_derivatives=overrides.get("enable_derivatives", True),
        auth_type=overrides.get("auth_type"),
        cdp_api_key=overrides.get("cdp_api_key"),
        cdp_private_key=overrides.get("cdp_private_key"),
    )
    return CoinbaseBrokerage(config)


class FakeReconnectTransport:
    """Transport that simulates a disconnect after a batch of messages."""

    def __init__(self, batches: list[tuple[list[dict[str, Any]], bool]]):
        self._batches: Deque[tuple[list[dict[str, Any]], bool]] = deque(batches)
        self.subscriptions: list[dict[str, Any]] = []
        self.connected = False

    def connect(self, url: str, headers: dict[str, Any] | None = None) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def subscribe(self, payload: dict[str, Any]) -> None:
        self.subscriptions.append(payload)

    def stream(self):
        if not self._batches:
            return iter(())
        messages, should_error = self._batches.popleft()
        yield from messages
        if should_error:
            raise RuntimeError("simulated disconnect")


def test_reconnect_and_resubscribe(monkeypatch):
    ws = CoinbaseWebSocket("wss://example", max_retries=3, base_delay=0)
    fake = FakeReconnectTransport(
        [
            ([{"seq": 1}, {"seq": 2}], True),
            ([{"seq": 3}], False),
        ]
    )
    ws.set_transport(fake)
    ws.subscribe(WSSubscription(channels=["market_trades"], product_ids=["BTC-USD"]))

    out = list(ws.stream_messages())
    assert [m["seq"] for m in out] == [1, 2, 3]
    assert len(fake.subscriptions) >= 2  # resubscribed after reconnect


class CaptureTransport:
    def __init__(self):
        self.connected = False
        self.subscriptions: list[dict[str, Any]] = []

    def connect(self, url: str, headers: dict[str, Any] | None = None) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def subscribe(self, payload: dict[str, Any]) -> None:
        self.subscriptions.append(payload)

    def stream(self):
        if False:
            yield {}


def test_ws_subscribe_includes_channels_and_products():
    transport = CaptureTransport()
    ws = CoinbaseWebSocket(url="wss://test", transport=transport)
    ws.connect()

    ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))
    payload = transport.subscriptions[0]
    assert payload["type"] == "subscribe"
    assert payload["channels"] == ["ticker"]
    assert payload["product_ids"] == ["BTC-USD"]


def test_ws_subscribe_handles_auth(monkeypatch):
    transport = CaptureTransport()
    ws = CoinbaseWebSocket(
        url="wss://test", transport=transport, ws_auth_provider=lambda: {"jwt": "token"}
    )
    ws.connect()

    ws.subscribe(WSSubscription(channels=["user"], product_ids=[]))
    assert transport.subscriptions[0]["jwt"] == "token"

    transport.subscriptions.clear()
    ws.subscribe(WSSubscription(channels=["user"], product_ids=[], auth_data={"jwt": "explicit"}))
    assert transport.subscriptions[0]["jwt"] == "explicit"


class TestMarketStreaming:
    def test_stream_trades_normalization(self):
        messages = [
            {
                "type": "trade",
                "product_id": "BTC-USD-PERP",
                "price": "50000.50",
                "size": "0.1",
                "time": "2024-01-15T12:00:00Z",
            },
            {
                "type": "trade",
                "product_id": "ETH-USD-PERP",
                "price": "3000.25",
                "size": "1.5",
                "time": "2024-01-15T12:00:01Z",
            },
            {
                "type": "trade",
                "product_id": "BTC-USD-PERP",
                "price": "50001.00",
                "size": "0.2",
                "time": "2024-01-15T12:00:02Z",
            },
        ]

        adapter = make_adapter()
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com", transport=MockTransport(messages=messages)
        )
        adapter._ws_factory_override = lambda: mock_ws

        trades = list(adapter.stream_trades(["BTC-USD-PERP", "ETH-USD-PERP"]))
        assert len(trades) == 3
        assert trades[0]["price"] == Decimal("50000.50")
        assert isinstance(trades[1]["size"], Decimal)

    def test_status_channel_subscription(self):
        adapter = make_adapter()
        subscriptions: list[dict[str, Any]] = []

        mock_transport = MockTransport(messages=[])
        original_subscribe = mock_transport.subscribe

        def track(payload: dict[str, Any]):
            subscriptions.append(payload)
            return original_subscribe(payload)

        mock_transport.subscribe = track
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com", transport=mock_transport
        )
        adapter._ws_factory_override = lambda: mock_ws

        sub = WSSubscription(channels=["status"], product_ids=["BTC-USD-PERP", "ETH-USD-PERP"])
        mock_ws.subscribe(sub)

        assert subscriptions[0]["channels"] == ["status"]
        assert subscriptions[0]["product_ids"] == ["BTC-USD-PERP", "ETH-USD-PERP"]

    def test_stream_orderbook_channel_selection(self):
        adapter = make_adapter()
        subscriptions: list[dict[str, Any]] = []

        mock_transport = MockTransport(messages=[])
        original_subscribe = mock_transport.subscribe

        def track(payload: dict[str, Any]):
            subscriptions.append(payload)
            return original_subscribe(payload)

        mock_transport.subscribe = track
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com", transport=mock_transport
        )
        adapter._ws_factory_override = lambda: mock_ws

        stream = adapter.stream_orderbook(["BTC-USD-PERP"], level=1)
        with pytest.raises(StopIteration):
            next(stream)
        assert subscriptions[0]["channels"] == ["ticker"]

        subscriptions.clear()
        mock_ws2 = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com", transport=MockTransport(messages=[])
        )
        mock_ws2._transport.subscribe = track
        adapter._ws_factory_override = lambda: mock_ws2

        stream = adapter.stream_orderbook(["ETH-USD-PERP"], level=2)
        with pytest.raises(StopIteration):
            next(stream)
        assert subscriptions[0]["channels"] == ["level2"]

    def test_normalize_market_message(self):
        msg = {
            "price": "12345.67",
            "size": "1.234",
            "best_bid": "12340.00",
            "best_ask": "12350.00",
            "volume": "100.5",
        }
        normalized = normalize_market_message(msg)
        assert normalized["price"] == Decimal("12345.67")
        assert normalized["volume"] == Decimal("100.5")

        msg_with_time = {"time": "2024-01-15T12:00:00Z", "price": "100"}
        normalized = normalize_market_message(msg_with_time)
        assert normalized["timestamp"] == "2024-01-15T12:00:00Z"

        msg_invalid = {"price": "invalid", "size": None, "volume": ""}
        normalized = normalize_market_message(msg_invalid)
        assert normalized["price"] == "invalid"


class TestAuthenticatedUserEvents:
    def test_stream_user_events_gap_detection(self):
        messages = [
            {"type": "order", "sequence": 100},
            {"type": "order", "sequence": 101},
            {"type": "order", "sequence": 103},
            {"type": "fill", "sequence": 104},
        ]
        adapter = make_adapter()
        mock_ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com", transport=MockTransport(messages=messages)
        )
        adapter._ws_factory_override = lambda: mock_ws

        events = list(adapter.stream_user_events())
        assert events[2]["gap_detected"] is True
        assert events[2]["last_seq"] == 101
        assert "gap_detected" not in events[3]

    def test_ws_auth_provider_injection(self):
        adapter = make_adapter(auth_type="JWT", cdp_api_key="k", cdp_private_key="secret")
        subscribe_payloads: list[dict[str, Any]] = []

        mock_transport = MockTransport(messages=[])
        original_subscribe = mock_transport.subscribe

        def track(payload: dict[str, Any]):
            subscribe_payloads.append(payload)
            return original_subscribe(payload)

        mock_transport.subscribe = track
        ws = CoinbaseWebSocket(
            url="wss://advanced-trade-ws.coinbase.com",
            transport=mock_transport,
            ws_auth_provider=lambda: {"jwt": "jwt_token"},
        )

        ws.subscribe(WSSubscription(channels=["user"], product_ids=[]))
        assert subscribe_payloads[0]["jwt"] == "jwt_token"


class TestReconnectAndLiveness:
    def test_sequence_guard_reset(self):
        guard = SequenceGuard()
        guard.annotate({"sequence": 100})
        guard.annotate({"sequence": 101})
        assert guard.last_seq == 101

        guard.reset()
        assert guard.last_seq is None
        guard.annotate({"sequence": 200})
        assert guard.last_seq == 200

    def test_ws_reconnect_sequence_guard(self):
        class ErrorAfterNTransport:
            def __init__(self, n: int = 2):
                self.n = n
                self.subscriptions: list[dict[str, Any]] = []
                self.stream_attempt = 0

            def connect(self, url: str, headers: dict[str, Any] | None = None) -> None:
                pass

            def disconnect(self) -> None:
                pass

            def subscribe(self, payload: dict[str, Any]) -> None:
                self.subscriptions.append(payload)

            def stream(self):
                self.stream_attempt += 1
                if self.stream_attempt == 1:
                    yield {"sequence": 1, "data": "msg1"}
                    raise ConnectionError("disconnect")
                yield {"sequence": 10, "data": "msg2"}
                yield {"sequence": 11, "data": "msg3"}

        ws = CoinbaseWebSocket(url="wss://test", max_retries=2, base_delay=0)
        transport = ErrorAfterNTransport()
        ws._transport = transport
        ws.subscribe(WSSubscription(channels=["user"], product_ids=["BTC-USD"]))

        messages = list(ws.stream_messages())
        assert [m["data"] for m in messages] == ["msg1", "msg2", "msg3"]
        assert len(transport.subscriptions) == 2

    def test_ws_factory_uses_endpoint_urls(self, monkeypatch):
        from bot_v2.features.brokerages.coinbase import adapter as adapter_mod

        created_urls: list[str] = []

        class StubWS:
            def __init__(self, url: str, **_):
                self.url = url
                created_urls.append(url)

            def connect(self):
                pass

            def disconnect(self):
                pass

            def subscribe(self, payload: dict[str, Any]):
                pass

            def stream_messages(self):
                if False:
                    yield {}

        monkeypatch.setattr(adapter_mod, "CoinbaseWebSocket", StubWS)

        adv = make_adapter()
        ws = adv._create_ws()
        assert ws.url == adv.endpoints.websocket_url()

        exch = make_adapter(
            base_url="https://api-public.sandbox.exchange.coinbase.com",
            sandbox=True,
            api_mode="exchange",
        )
        ws2 = exch._create_ws()
        assert ws2.url == exch.endpoints.websocket_url()


def test_user_stream_gap_detection(monkeypatch):
    class DummyWS:
        def __init__(self, messages: list[dict[str, Any]]):
            self.messages = deque(messages)
            self.subscriptions: list[WSSubscription] = []

        def subscribe(self, subscription: WSSubscription) -> None:
            self.subscriptions.append(subscription)

        def stream_messages(self):
            while self.messages:
                yield self.messages.popleft()

    dummy_ws = DummyWS(
        [
            {"sequence": 1, "type": "order"},
            {"sequence": 2, "type": "fill"},
            {"sequence": 4, "type": "fill"},
        ]
    )

    def fake_factory(*_, **__):
        return dummy_ws

    broker = CoinbaseBrokerage(
        APIConfig(api_key="k", api_secret="s", passphrase=None, base_url="https://api")
    )
    monkeypatch.setattr(CoinbaseBrokerage, "_create_ws", fake_factory)

    out = list(broker.stream_user_events())
    assert out[-1]["gap_detected"] is True
    assert out[-1]["last_seq"] == 2
    assert len(dummy_ws.subscriptions) == 1


class TestMultiChannelSubscriptions:
    def test_multiple_channels_single_payload(self):
        subs: list[dict[str, Any]] = []
        transport = MockTransport(messages=[])
        original = transport.subscribe

        def track(payload: dict[str, Any]):
            subs.append(payload)
            return original(payload)

        transport.subscribe = track

        ws = CoinbaseWebSocket(url="wss://test", transport=transport)
        ws.connect()
        ws.subscribe(
            WSSubscription(channels=["ticker", "level2"], product_ids=["BTC-USD", "ETH-USD"])
        )
        assert subs[0]["channels"] == ["ticker", "level2"]

    def test_multiple_subscribe_calls(self):
        subs: list[dict[str, Any]] = []
        transport = MockTransport(messages=[])
        original = transport.subscribe

        def track(payload: dict[str, Any]):
            subs.append(payload)
            return original(payload)

        transport.subscribe = track
        ws = CoinbaseWebSocket(url="wss://test", transport=transport)
        ws.connect()
        ws.subscribe(WSSubscription(channels=["ticker"], product_ids=["BTC-USD"]))
        ws.subscribe(WSSubscription(channels=["trades"], product_ids=["BTC-USD"]))
        assert [p["channels"] for p in subs] == [["ticker"], ["trades"]]


def test_liveness_timeout_configured():
    ws = CoinbaseWebSocket(url="wss://test", liveness_timeout=0.1, max_retries=1, base_delay=0.01)
    assert ws._liveness_timeout == 0.1
