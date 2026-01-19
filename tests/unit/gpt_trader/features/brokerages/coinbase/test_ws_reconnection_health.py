"""Tests for CoinbaseWebSocket health fields related to reconnection."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket


class TestWebSocketHealthIncludesNewFields:
    """Tests that get_health includes the new reconnection fields."""

    def test_health_includes_connected_since(self) -> None:
        """Test that health dict includes connected_since field."""
        ws = CoinbaseWebSocket()
        ws._connected_since = 12345.0

        health = ws.get_health()

        assert "connected_since" in health
        assert health["connected_since"] == 12345.0

    def test_health_includes_max_attempts_triggered(self) -> None:
        """Test that health dict includes max_attempts_triggered field."""
        ws = CoinbaseWebSocket()
        ws._max_attempts_triggered = True

        health = ws.get_health()

        assert "max_attempts_triggered" in health
        assert health["max_attempts_triggered"] is True
