"""Tests for CoinbaseWebSocket.get_health()."""

from __future__ import annotations


class TestCoinbaseWebSocketHealth:
    """Tests for CoinbaseWebSocket.get_health()."""

    def test_get_health_returns_all_fields(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()

        health = ws.get_health()

        assert "connected" in health
        assert "last_message_ts" in health
        assert "last_heartbeat_ts" in health
        assert "last_close_ts" in health
        assert "last_error_ts" in health
        assert "gap_count" in health
        assert "reconnect_count" in health

    def test_get_health_initial_state(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()

        health = ws.get_health()

        assert health["connected"] is False
        assert health["last_message_ts"] is None
        assert health["last_heartbeat_ts"] is None
        assert health["last_close_ts"] is None
        assert health["last_error_ts"] is None
        assert health["gap_count"] == 0
        assert health["reconnect_count"] == 0

    def test_get_health_reflects_gap_count(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        ws._gap_count = 5

        health = ws.get_health()

        assert health["gap_count"] == 5

    def test_get_health_reflects_reconnect_count(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import CoinbaseWebSocket

        ws = CoinbaseWebSocket()
        ws._reconnect_count = 3

        health = ws.get_health()

        assert health["reconnect_count"] == 3
