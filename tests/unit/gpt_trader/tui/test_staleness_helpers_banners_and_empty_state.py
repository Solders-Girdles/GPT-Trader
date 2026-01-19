"""Tests for staleness banners and empty-state helpers."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from gpt_trader.tui.staleness_helpers import (
    get_connection_banner,
    get_empty_state_config,
    get_staleness_banner,
)


class TestGetStalenessBanner:
    """Tests for get_staleness_banner function."""

    def test_returns_reconnecting_when_fetching(self) -> None:
        """Should return reconnecting banner when data_fetching is True."""
        state = MagicMock()
        state.data_fetching = True

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "Reconnecting" in text
        assert severity == "info"

    def test_returns_degraded_banner(self) -> None:
        """Should return degraded banner when in degraded mode."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = True
        state.degraded_reason = "Test reason"

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "Degraded" in text
        assert "Test reason" in text
        assert severity == "warning"

    def test_returns_none_when_fresh(self) -> None:
        """Should return None when data is fresh."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 5
        # Execution health: healthy (no executions)
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 0

        assert get_staleness_banner(state) is None

    def test_returns_warning_when_stale(self) -> None:
        """Should return warning when data is stale."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 20
        # Execution health: healthy (no executions)
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 0

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "stale" in text.lower()
        assert severity == "warning"

    def test_returns_error_when_critical(self) -> None:
        """Should return error with reconnect hint when critical."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 45
        # Execution health: healthy (no executions)
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 0

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "stale" in text.lower()
        assert "R" in text  # Reconnect hint
        assert severity == "error"


class TestGetConnectionBanner:
    """Tests for get_connection_banner function."""

    def test_returns_none_when_stopped(self) -> None:
        """Should return None when bot is stopped."""
        result = get_connection_banner(
            connection_status="UNKNOWN",
            bot_running=False,
            degraded_mode=False,
        )
        assert result is None

    def test_returns_none_for_transitional_states(self) -> None:
        """Should return None for connecting/unknown states."""
        for status in ["CONNECTING", "RECONNECTING", "SYNCING", "UNKNOWN", "--", ""]:
            result = get_connection_banner(
                connection_status=status,
                bot_running=True,
                degraded_mode=False,
            )
            assert result is None, f"Expected None for status {status}"

    def test_returns_none_for_healthy_states(self) -> None:
        """Should return None for healthy connection states."""
        for status in ["CONNECTED", "OK", "HEALTHY"]:
            result = get_connection_banner(
                connection_status=status,
                bot_running=True,
                degraded_mode=False,
            )
            assert result is None, f"Expected None for status {status}"

    def test_returns_error_for_disconnected(self) -> None:
        """Should return error banner for disconnected state."""
        result = get_connection_banner(
            connection_status="DISCONNECTED",
            bot_running=True,
            degraded_mode=False,
        )
        assert result is not None
        text, severity = result
        assert "Connection lost" in text
        assert severity == "error"


class TestGetEmptyStateConfig:
    """Tests for get_empty_state_config function."""

    def test_returns_stopped_config_when_not_running(self) -> None:
        """Should return stopped config when bot is not running."""
        config = get_empty_state_config(
            data_type="Market",
            bot_running=False,
            data_source_mode="paper",
            connection_status="",
        )
        assert "Awaiting Start" in config["title"]
        assert "[S] Start Bot" in config["actions"]

    def test_returns_read_only_config(self) -> None:
        """Should return read-only config for read_only mode."""
        config = get_empty_state_config(
            data_type="Market",
            bot_running=False,
            data_source_mode="read_only",
            connection_status="",
        )
        assert "Feed Paused" in config["title"]
        assert "[S] Start Feed" in config["actions"]

    def test_returns_connection_failed_config(self) -> None:
        """Should return connection failed config for error states."""
        config = get_empty_state_config(
            data_type="Market",
            bot_running=True,
            data_source_mode="live",
            connection_status="DISCONNECTED",
        )
        assert "Connection Failed" in config["title"]
        assert "[R] Reconnect" in config["actions"]

    def test_returns_connecting_config(self) -> None:
        """Should return connecting config for transitional states."""
        config = get_empty_state_config(
            data_type="Market",
            bot_running=True,
            data_source_mode="live",
            connection_status="CONNECTING",
        )
        assert "Connecting" in config["title"]
        assert config["actions"] == []

    def test_returns_no_data_config(self) -> None:
        """Should return no data config when running but no data."""
        config = get_empty_state_config(
            data_type="Strategy",
            bot_running=True,
            data_source_mode="live",
            connection_status="CONNECTED",
        )
        assert "No Strategy Yet" in config["title"]
        assert "[R] Refresh" in config["actions"]
