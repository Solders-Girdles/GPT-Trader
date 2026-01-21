"""Tests for Coinbase WebSocket event types, SequenceGuard, and health monitoring."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.ws_events import EventType


class TestEventType:
    """Tests for EventType enum."""

    def test_event_types_values(self) -> None:
        assert EventType.TICKER.value == "ticker"
        assert EventType.LEVEL2.value == "l2_data"
        assert EventType.USER.value == "user"
        assert EventType.ERROR.value == "error"


class TestSequenceGuard:
    """Tests for SequenceGuard gap detection."""

    def test_annotate_returns_message_unchanged_without_sequence(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()
        message = {"type": "ticker", "price": "50000"}

        result = guard.annotate(message)

        assert result == message
        assert "gap_detected" not in result

    def test_annotate_first_message_no_gap(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()
        message = {"sequence": 1, "type": "ticker"}

        result = guard.annotate(message)

        assert "gap_detected" not in result

    def test_annotate_sequential_messages_no_gap(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()

        msg1 = guard.annotate({"sequence": 1})
        msg2 = guard.annotate({"sequence": 2})
        msg3 = guard.annotate({"sequence": 3})

        assert "gap_detected" not in msg1
        assert "gap_detected" not in msg2
        assert "gap_detected" not in msg3

    def test_annotate_gap_detected(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()

        guard.annotate({"sequence": 1})
        result = guard.annotate({"sequence": 5})  # Gap: 2, 3, 4 missing

        assert result.get("gap_detected") is True

    def test_reset_clears_state(self) -> None:
        from gpt_trader.features.brokerages.coinbase.ws import SequenceGuard

        guard = SequenceGuard()

        guard.annotate({"sequence": 100})
        guard.reset()

        # After reset, first message should not detect gap
        result = guard.annotate({"sequence": 1})
        assert "gap_detected" not in result


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
