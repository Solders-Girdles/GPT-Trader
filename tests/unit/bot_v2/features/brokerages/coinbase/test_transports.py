from __future__ import annotations

import json
import sys
import types
from collections import deque

import pytest

from bot_v2.features.brokerages.coinbase import transports


class DummySocket:
    def __init__(self, messages: list[str]) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._messages = deque(messages)

    def send(self, message: str) -> None:
        self.sent.append(message)

    def recv(self) -> str:
        if self._messages:
            return self._messages.popleft()
        raise RuntimeError("stream end")

    def close(self) -> None:
        self.closed = True


def _install_websocket(monkeypatch: pytest.MonkeyPatch, socket: DummySocket) -> None:
    module = types.SimpleNamespace(create_connection=lambda url, header=None: socket)
    monkeypatch.setitem(sys.modules, "websocket", module)


def test_real_transport_connect_and_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = DummySocket([json.dumps({"type": "snapshot"})])
    _install_websocket(monkeypatch, socket)

    transport = transports.RealTransport()
    transport.connect("wss://example", headers={"Auth": "token"})
    transport.subscribe({"type": "subscribe", "product_ids": ["BTC-USD"]})

    stream = transport.stream()
    first = next(stream)
    assert first["type"] == "snapshot"
    with pytest.raises(RuntimeError):
        next(stream)

    transport.disconnect()
    assert socket.closed


def test_real_transport_requires_connection() -> None:
    transport = transports.RealTransport()
    with pytest.raises(RuntimeError):
        transport.subscribe({"type": "ping"})
    with pytest.raises(RuntimeError):
        next(transport.stream())


def test_mock_transport_records_calls() -> None:
    transport = transports.MockTransport()
    transport.connect("wss://mock", headers={"Auth": "1"})
    transport.subscribe({"type": "subscribe"})
    transport.add_message({"type": "update"})

    stream = list(transport.stream())
    assert transport.connected
    assert transport.subscriptions[0]["type"] == "subscribe"
    assert stream[-1]["type"] == "update"


def test_noop_transport_behaviour(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    transport = transports.NoopTransport()
    transport.connect("wss://none")
    transport.subscribe({"type": "anything"})
    transport.disconnect()
    assert transport.connected is False
    assert any("NoopTransport" in record.message for record in caplog.records)
