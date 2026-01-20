"""Tests for WebSocket reconnection degradation callbacks."""

from __future__ import annotations

import pytest

import gpt_trader.features.brokerages.coinbase.ws as ws_module
from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket


class TestWebSocketDegradationCallback:
    """Tests for degradation callback when max attempts exceeded."""

    def test_callback_triggered_on_max_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that degradation callback is triggered when max attempts exceeded."""
        callback_called = []

        def on_max_exceeded(pause_seconds: int) -> None:
            callback_called.append(pause_seconds)

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=on_max_exceeded)

        # Set up to exceed max attempts
        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        monkeypatch.setattr(ws_module, "WS_RECONNECT_PAUSE_SECONDS", 300)
        ws._reconnect_count = 3  # At max
        ws._running.set()

        # Trigger close (will exceed max on increment)
        ws._on_close(None, 1000, "Connection lost")

        # Callback should have been called
        assert len(callback_called) == 1
        assert callback_called[0] == 300

    def test_callback_only_triggered_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that degradation callback is only triggered once per max-exceeded event."""
        callback_count = []

        def on_max_exceeded(pause_seconds: int) -> None:
            callback_count.append(1)

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=on_max_exceeded)

        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        ws._reconnect_count = 3
        ws._running.set()

        # First close - should trigger callback
        ws._on_close(None, 1000, "Connection lost")

        # Second close - should NOT trigger callback again
        # Note: After max attempts triggered, _running is cleared
        # so we need to set it again to simulate another close
        ws._running.set()
        ws._shutdown.set()  # Prevent actual reconnection attempt
        ws._on_close(None, 1000, "Connection lost again")

        # Callback should only be called once
        assert len(callback_count) == 1

    def test_no_callback_when_none_provided(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that missing callback doesn't cause errors."""
        ws = CoinbaseWebSocket(on_max_attempts_exceeded=None)

        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        ws._reconnect_count = 3
        ws._running.set()

        # Should not raise even without callback
        ws._on_close(None, 1000, "Connection lost")

        # Verify max_attempts_triggered flag is set
        assert ws._max_attempts_triggered is True

    def test_callback_exception_does_not_crash(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that exception in callback is handled gracefully."""

        def bad_callback(pause_seconds: int) -> None:
            raise ValueError("Intentional test error")

        ws = CoinbaseWebSocket(on_max_attempts_exceeded=bad_callback)

        monkeypatch.setattr(ws_module, "WS_RECONNECT_MAX_ATTEMPTS", 3)
        ws._reconnect_count = 3
        ws._running.set()

        # Should not raise even with bad callback
        ws._on_close(None, 1000, "Connection lost")

        # Should still set the triggered flag
        assert ws._max_attempts_triggered is True
