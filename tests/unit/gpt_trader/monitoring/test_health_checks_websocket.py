"""Tests for the websocket freshness health check."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.monitoring.health_checks import check_ws_freshness
from gpt_trader.utilities.time_provider import FakeClock


class TestCheckWsFreshness:
    """Tests for check_ws_freshness function."""

    def test_broker_without_ws_support(self) -> None:
        """Test handling when broker doesn't support WS health."""
        broker = MagicMock(spec=["list_balances"])  # No get_ws_health

        healthy, details = check_ws_freshness(broker)

        assert healthy is True
        assert details.get("ws_not_supported") is True

    def test_ws_not_initialized(self) -> None:
        """Test handling when WS not initialized (returns None)."""
        broker = MagicMock()
        broker.get_ws_health.return_value = None

        healthy, details = check_ws_freshness(broker)

        assert healthy is True
        assert details.get("ws_not_initialized") is True

    def test_ws_connected_and_fresh(self) -> None:
        """Test healthy WS with fresh messages."""
        import time

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 5,  # 5 seconds ago
            "last_heartbeat_ts": time.time() - 10,  # 10 seconds ago
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is True
        assert details["connected"] is True
        assert details["stale"] is False

    def test_ws_stale_message(self) -> None:
        """Test failure when messages are stale."""
        import time

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 120,  # 2 minutes ago (stale)
            "last_heartbeat_ts": time.time() - 30,
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker, message_stale_seconds=60.0)

        assert healthy is False
        assert details["stale"] is True
        assert details["stale_reason"] == "message"

    def test_ws_disconnected(self) -> None:
        """Test failure when WS is disconnected."""
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": False,
            "last_message_ts": 0,
            "last_heartbeat_ts": 0,
            "gap_count": 0,
            "reconnect_count": 5,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is False
        assert details["connected"] is False

    def test_ws_max_attempts_triggered(self) -> None:
        """Test critical failure when max reconnect attempts triggered."""
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": False,
            "last_message_ts": 0,
            "last_heartbeat_ts": 0,
            "gap_count": 0,
            "reconnect_count": 10,
            "max_attempts_triggered": True,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is False
        assert details["max_attempts_triggered"] is True
        assert details["severity"] == "critical"

    def test_ws_freshness_uses_time_provider(self) -> None:
        """Test freshness checks use the injected time provider."""
        clock = FakeClock(start_time=1000.0)
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": 995.0,
            "last_heartbeat_ts": 990.0,
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(
            broker,
            message_stale_seconds=10.0,
            heartbeat_stale_seconds=15.0,
            time_provider=clock,
        )

        assert healthy is True
        assert details["stale"] is False

        clock.advance(20.0)
        healthy, details = check_ws_freshness(
            broker,
            message_stale_seconds=10.0,
            heartbeat_stale_seconds=15.0,
            time_provider=clock,
        )

        assert healthy is False
        assert details["stale"] is True
        assert details["stale_reason"] == "message"

    def test_ws_connected_with_unparseable_message_timestamp(self) -> None:
        """Connected WS should fail closed when message timestamp is invalid."""
        import time

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": {"invalid": "timestamp"},
            "last_heartbeat_ts": time.time() - 5,
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is False
        assert details["stale"] is True
        assert details["stale_reason"] == "message_timestamp_unparseable"

    def test_ws_connected_with_unparseable_heartbeat_timestamp(self) -> None:
        """Connected WS should fail closed when heartbeat timestamp is invalid."""
        import time

        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": time.time() - 5,
            "last_heartbeat_ts": "not-a-timestamp",
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is False
        assert details["stale"] is True
        assert details["stale_reason"] == "heartbeat_timestamp_unparseable"

    def test_ws_connected_with_missing_timestamps_keeps_prior_behavior(self) -> None:
        """Connected WS without timestamp values keeps legacy missing-data behavior."""
        broker = MagicMock()
        broker.get_ws_health.return_value = {
            "connected": True,
            "last_message_ts": 0,
            "last_heartbeat_ts": 0,
            "gap_count": 0,
            "reconnect_count": 0,
            "max_attempts_triggered": False,
        }

        healthy, details = check_ws_freshness(broker)

        assert healthy is True
        assert details["stale"] is False
