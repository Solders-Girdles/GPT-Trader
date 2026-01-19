"""Tests for WebSocket reconnection reset behavior."""

from __future__ import annotations

import time
from unittest.mock import patch

from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket


class TestWebSocketReconnectionReset:
    """Tests for reconnection attempt counter reset after stable connection."""

    def test_attempts_reset_after_stable_period(self) -> None:
        """Test that reconnect counter resets after stable connection."""
        ws = CoinbaseWebSocket()

        # Simulate some failed reconnection attempts
        ws._reconnect_count = 5
        ws._running.set()

        # Simulate a connection that was stable for 120 seconds
        ws._connected_since = time.time() - 120.0

        # Patch WS_RECONNECT_RESET_SECONDS to 60 for test
        with patch(
            "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_RESET_SECONDS",
            60.0,
        ):
            # Simulate _on_close being called (connection dropped)
            # Don't actually reconnect - just test the reset logic
            ws._shutdown.set()  # Prevent actual reconnection
            ws._on_close(None, 1000, "Normal closure")

        # Counter should be reset because connection was stable
        assert ws._reconnect_count == 0

    def test_attempts_not_reset_if_unstable(self) -> None:
        """Test that reconnect counter is NOT reset if connection was short-lived."""
        ws = CoinbaseWebSocket()

        # Simulate some failed reconnection attempts
        ws._reconnect_count = 5
        ws._running.set()

        # Simulate a connection that was only stable for 10 seconds
        ws._connected_since = time.time() - 10.0

        # Patch WS_RECONNECT_RESET_SECONDS to 60 for test
        with patch(
            "gpt_trader.features.brokerages.coinbase.ws.WS_RECONNECT_RESET_SECONDS",
            60.0,
        ):
            # Simulate _on_close being called
            ws._shutdown.set()  # Prevent actual reconnection
            ws._on_close(None, 1000, "Normal closure")

        # Counter should NOT be reset (connection wasn't stable long enough)
        # Note: counter increments by 1 because of the close event
        assert ws._reconnect_count == 5  # Stays at 5 (no increment since shutdown)
