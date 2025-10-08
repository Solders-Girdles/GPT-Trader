"""Comprehensive Coinbase websocket coverage."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Any, Deque
from collections.abc import Iterable

import pytest

from bot_v2.features.brokerages.coinbase.transports import MockTransport
from bot_v2.features.brokerages.coinbase.ws import (
    CoinbaseWebSocket,
    SequenceGuard,
    WSSubscription,
    normalize_market_message,
)

from tests.unit.bot_v2.features.brokerages.coinbase.websocket_test_utils import make_adapter


# --- Connectivity and reconnection behaviour -------------------------------------------------


class FakeReconnectTransport:
    """Transport that simulates reconnect cycles after streaming batches."""

    def __init__(self, batches: Iterable[tuple[list[dict[str, Any]], bool]]):
        self._batches: Deque[tuple[list[dict[str, Any]], bool]] = deque(batches)
        self.subscriptions: list[dict[str, Any]] = []

    def connect(self, url: str, headers: dict[str, Any] | None = None) -> None:
        del url, headers

    def disconnect(self) -> None:  # pragma: no cover - handled via higher level
        return

    def subscribe(self, payload: dict[str, Any]) -> None:
        self.subscriptions.append(payload)

    def stream(self):
        if not self._batches:
            return iter(())
        messages, should_error = self._batches.popleft()
        yield from messages
        if should_error:
            raise RuntimeError("simulated disconnect")


def test_reconnect_and_resubscribe() -> None:
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
    assert len(fake.subscriptions) >= 2


def test_sequence_guard_reset() -> None:
    guard = SequenceGuard()
    guard.annotate({"sequence": 100})
    guard.annotate({"sequence": 101})
    assert guard.last_seq == 101

    guard.reset()
    assert guard.last_seq is None
    guard.annotate({"sequence": 200})
    assert guard.last_seq == 200


def test_ws_reconnect_sequence_guard() -> None:
    class ErrorAfterNTransport:
        def __init__(self, n: int = 2):
            self.n = n
            self.subscriptions: list[dict[str, Any]] = []
            self.stream_attempt = 0

        def connect(self, url: str, headers: dict[str, Any] | None = None) -> None:
            del url, headers

        def disconnect(self) -> None:  # pragma: no cover - helper stub
            return

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
    ws.set_transport(transport)
    ws.subscribe(WSSubscription(channels=["user"], product_ids=["BTC-USD"]))

    messages = list(ws.stream_messages())
    assert [m["data"] for m in messages] == ["msg1", "msg2", "msg3"]
    assert len(transport.subscriptions) == 2


def test_ws_factory_uses_endpoint_urls(monkeypatch) -> None:
    from bot_v2.features.brokerages.coinbase import adapter as adapter_mod

    created_urls: list[str] = []

    class StubWS:
        def __init__(self, url: str, **_):
            self.url = url
            created_urls.append(url)

        def connect(self) -> None:  # pragma: no cover - helper stub
            return

        def disconnect(self) -> None:  # pragma: no cover - helper stub
            return

        def subscribe(self, payload: dict[str, Any]) -> None:  # pragma: no cover - helper stub
            del payload

        def stream_messages(self):  # pragma: no cover - helper stub
            from bot_v2.utilities import empty_stream

            return empty_stream()

    monkeypatch.setattr(adapter_mod, "CoinbaseWebSocket", StubWS)

    advanced = make_adapter()
    ws = advanced._create_ws()
    assert ws.url == advanced.endpoints.websocket_url()

    exchange = make_adapter(
        base_url="https://api-public.sandbox.exchange.coinbase.com",
        sandbox=True,
        api_mode="exchange",
    )
    ws2 = exchange._create_ws()
    assert ws2.url == exchange.endpoints.websocket_url()


# --- Streaming and normalization flows -------------------------------------------------------


def test_stream_trades_normalization() -> None:
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


MARKET_MESSAGE_CASES = [
    pytest.param(
        {
            "id": "decimal-fields",
            "message": {
                "price": "12345.67",
                "size": "1.234",
                "best_bid": "12340.00",
                "best_ask": "12350.00",
                "volume": "100.5",
            },
            "expected": {
                "price": Decimal("12345.67"),
                "size": Decimal("1.234"),
                "best_bid": Decimal("12340.00"),
                "best_ask": Decimal("12350.00"),
                "volume": Decimal("100.5"),
            },
        },
        id="decimal-fields",
    ),
    pytest.param(
        {
            "id": "timestamp-normalization",
            "message": {"time": "2024-01-15T12:00:00Z", "price": "100"},
            "expected": {"timestamp": "2024-01-15T12:00:00Z"},
        },
        id="timestamp-normalization",
    ),
    pytest.param(
        {
            "id": "invalid-handling",
            "message": {"price": "invalid", "size": None, "volume": ""},
            "expected": {"price": "invalid"},
        },
        id="invalid-handling",
    ),
]


@pytest.mark.parametrize("case", MARKET_MESSAGE_CASES, ids=lambda c: c["id"])
def test_normalize_market_message(case: dict[str, Any]) -> None:
    normalized = normalize_market_message(case["message"])
    for key, expected_value in case["expected"].items():
        assert normalized[key] == expected_value


def test_stream_user_events_gap_detection() -> None:
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


def test_user_stream_gap_detection(monkeypatch) -> None:
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

    broker = make_adapter(base_url="https://api")
    monkeypatch.setattr(type(broker), "_create_ws", lambda *_: dummy_ws)

    out = list(broker.stream_user_events())
    assert out[-1]["gap_detected"] is True
    assert out[-1]["last_seq"] == 2
    assert len(dummy_ws.subscriptions) == 1


# --- Subscription payload composition -------------------------------------------------------


class CaptureTransport:
    def __init__(self) -> None:
        self.connected = False
        self.subscriptions: list[dict[str, Any]] = []

    def connect(self, url: str, headers: dict[str, Any] | None = None) -> None:
        del url, headers
        self.connected = True

    def disconnect(self) -> None:  # pragma: no cover - helper stub
        self.connected = False

    def subscribe(self, payload: dict[str, Any]) -> None:
        self.subscriptions.append(payload)

    def stream(self):  # pragma: no cover - helper stub
        from bot_v2.utilities import empty_stream

        return empty_stream()


SUBSCRIPTION_CASES = [
    pytest.param(
        {
            "id": "basic",
            "channels": ["ticker"],
            "product_ids": ["BTC-USD"],
            "provider_auth": None,
            "auth_data": None,
            "expected_auth": {},
        },
        id="basic",
    ),
    pytest.param(
        {
            "id": "provider-auth",
            "channels": ["user"],
            "product_ids": [],
            "provider_auth": {"jwt": "provider-token"},
            "auth_data": None,
            "expected_auth": {"jwt": "provider-token"},
        },
        id="provider-auth",
    ),
    pytest.param(
        {
            "id": "explicit-auth",
            "channels": ["user"],
            "product_ids": [],
            "provider_auth": {"jwt": "provider-token"},
            "auth_data": {"jwt": "explicit-token"},
            "expected_auth": {"jwt": "explicit-token"},
        },
        id="explicit-auth",
    ),
]


@pytest.mark.parametrize("case", SUBSCRIPTION_CASES, ids=lambda c: c["id"])
def test_ws_subscription_payloads(case: dict[str, Any]) -> None:
    transport = CaptureTransport()
    provider_data = case.get("provider_auth")
    provider = (lambda data=provider_data: data.copy()) if provider_data else None
    ws = CoinbaseWebSocket(url="wss://test", transport=transport, ws_auth_provider=provider)
    ws.connect()

    subscription = WSSubscription(
        channels=case["channels"],
        product_ids=case["product_ids"],
        auth_data=case.get("auth_data"),
    )
    ws.subscribe(subscription)

    payload = transport.subscriptions[0]
    assert payload["type"] == "subscribe"
    assert payload["channels"] == case["channels"]
    assert payload["product_ids"] == case["product_ids"]

    expected_auth = case.get("expected_auth") or {}
    for key, expected_value in expected_auth.items():
        assert payload[key] == expected_value
    if not expected_auth:
        assert "jwt" not in payload


@pytest.mark.parametrize(
    "level, expected_channel",
    [
        pytest.param(1, "ticker", id="level-1"),
        pytest.param(2, "level2", id="level-2"),
    ],
)
def test_stream_orderbook_channel_selection(level: int, expected_channel: str) -> None:
    adapter = make_adapter()
    subscriptions: list[dict[str, Any]] = []

    mock_transport = MockTransport(messages=[])
    original_subscribe = mock_transport.subscribe

    def track(payload: dict[str, Any]) -> None:
        subscriptions.append(payload)
        original_subscribe(payload)

    mock_transport.subscribe = track
    mock_ws = CoinbaseWebSocket(
        url="wss://advanced-trade-ws.coinbase.com", transport=mock_transport
    )
    adapter._ws_factory_override = lambda: mock_ws

    stream = adapter.stream_orderbook(["BTC-USD-PERP"], level=level)
    with pytest.raises(StopIteration):
        next(stream)

    assert subscriptions[0]["channels"] == [expected_channel]


def test_status_channel_subscription() -> None:
    adapter = make_adapter()
    subscriptions: list[dict[str, Any]] = []

    mock_transport = MockTransport(messages=[])
    original_subscribe = mock_transport.subscribe

    def track(payload: dict[str, Any]) -> None:
        subscriptions.append(payload)
        original_subscribe(payload)

    mock_transport.subscribe = track
    mock_ws = CoinbaseWebSocket(
        url="wss://advanced-trade-ws.coinbase.com", transport=mock_transport
    )
    adapter._ws_factory_override = lambda: mock_ws

    sub = WSSubscription(channels=["status"], product_ids=["BTC-USD-PERP", "ETH-USD-PERP"])
    mock_ws.subscribe(sub)

    payload = subscriptions[0]
    assert payload["channels"] == ["status"]
    assert payload["product_ids"] == ["BTC-USD-PERP", "ETH-USD-PERP"]


def test_ws_auth_provider_injection() -> None:
    subscribe_payloads: list[dict[str, Any]] = []
    mock_transport = MockTransport(messages=[])
    original_subscribe = mock_transport.subscribe

    def track(payload: dict[str, Any]) -> None:
        subscribe_payloads.append(payload)
        original_subscribe(payload)

    mock_transport.subscribe = track
    ws = CoinbaseWebSocket(
        url="wss://advanced-trade-ws.coinbase.com",
        transport=mock_transport,
        ws_auth_provider=lambda: {"jwt": "jwt_token"},
    )

    ws.subscribe(WSSubscription(channels=["user"], product_ids=[]))
    assert subscribe_payloads[0]["jwt"] == "jwt_token"
