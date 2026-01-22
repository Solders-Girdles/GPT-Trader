from __future__ import annotations

import time

import pytest

import gpt_trader.features.brokerages.coinbase.ws as ws_module
from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket


class TestWebSocketDegradationCallback:
    """Tests for degradation callback when max attempts exceeded."""

    def test_callback_triggered_on_max_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        callback_called = []

        def on_max_exceeded(pause_seconds: int) -> None:
            callback_called.append(pause_seconds)

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=on_max_exceeded)

        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(ws_module, "WS_RECONNECT_PAUSE_SECONDS", 300)
        ws._reconnect_count = 3
        ws._running.set()

        ws._on_close(None, 1000, "Connection lost")

        assert len(callback_called) == 1
        assert callback_called[0] == 300

    def test_callback_only_triggered_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        callback_count = []

        def on_max_exceeded(pause_seconds: int) -> None:
            callback_count.append(1)

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=on_max_exceeded)

        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        ws._reconnect_count = 3
        ws._running.set()

        ws._on_close(None, 1000, "Connection lost")

        ws._running.set()
        ws._shutdown.set()
        ws._on_close(None, 1000, "Connection lost again")

        assert len(callback_count) == 1

    def test_no_callback_when_none_provided(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ws = CoinbaseWebSocket(on_max_attempts_exceeded=None)

        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        ws._reconnect_count = 3
        ws._running.set()

        ws._on_close(None, 1000, "Connection lost")

        assert ws._max_attempts_triggered is True

    def test_callback_exception_does_not_crash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def bad_callback(pause_seconds: int) -> None:
            raise ValueError("Intentional test error")

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=bad_callback)

        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        ws._reconnect_count = 3
        ws._running.set()

        ws._on_close(None, 1000, "Connection lost")

        assert ws._max_attempts_triggered is True


class TestWebSocketHealthIncludesNewFields:
    """Tests that get_health includes reconnection fields."""

    def test_health_includes_connected_since(self) -> None:
        ws = CoinbaseWebSocket()
        ws._connected_since = 12345.0

        health = ws.get_health()

        assert "connected_since" in health
        assert health["connected_since"] == 12345.0

    def test_health_includes_max_attempts_triggered(self) -> None:
        ws = CoinbaseWebSocket()
        ws._max_attempts_triggered = True

        health = ws.get_health()

        assert "max_attempts_triggered" in health
        assert health["max_attempts_triggered"] is True


class TestWebSocketReconnectionReset:
    """Tests for reconnection attempt counter reset after stable connection."""

    def test_attempts_reset_after_stable_period(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = CoinbaseWebSocket()
        ws._reconnect_count = 5
        ws._running.set()
        ws._connected_since = time.time() - 120.0

        monkeypatch.setattr(ws_module, "WS_RECONNECT_RESET_SECONDS", 60.0)
        ws._shutdown.set()
        ws._on_close(None, 1000, "Normal closure")

        assert ws._reconnect_count == 0

    def test_attempts_not_reset_if_unstable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = CoinbaseWebSocket()
        ws._reconnect_count = 5
        ws._running.set()
        ws._connected_since = time.time() - 10.0

        monkeypatch.setattr(ws_module, "WS_RECONNECT_RESET_SECONDS", 60.0)
        ws._shutdown.set()
        ws._on_close(None, 1000, "Normal closure")

        assert ws._reconnect_count == 5
