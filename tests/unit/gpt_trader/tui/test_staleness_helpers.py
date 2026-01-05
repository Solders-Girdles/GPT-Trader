"""Tests for staleness_helpers module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.staleness_helpers import (
    FRESH_THRESHOLD_SECONDS,
    STALE_THRESHOLD_SECONDS,
    format_freshness_label,
    get_connection_banner,
    get_data_age_seconds,
    get_empty_state_config,
    get_freshness_display,
    get_staleness_banner,
    get_staleness_severity,
)


class TestGetDataAgeSeconds:
    """Tests for get_data_age_seconds function."""

    def test_returns_none_when_no_data_available(self) -> None:
        """Should return None when data_available is False."""
        state = MagicMock()
        state.data_available = False
        state.last_data_fetch = time.time()

        assert get_data_age_seconds(state) is None

    def test_returns_none_when_last_fetch_zero(self) -> None:
        """Should return None when last_data_fetch is 0."""
        state = MagicMock()
        state.data_available = True
        state.last_data_fetch = 0

        assert get_data_age_seconds(state) is None

    def test_returns_age_when_data_available(self) -> None:
        """Should return age in seconds when data is available."""
        state = MagicMock()
        state.data_available = True
        state.last_data_fetch = time.time() - 15

        age = get_data_age_seconds(state)
        assert age is not None
        assert 14 <= age <= 16  # Allow small timing variance


class TestGetStalenessSeverity:
    """Tests for get_staleness_severity function."""

    def test_returns_critical_when_connection_unhealthy(self) -> None:
        """Should return critical when connection_healthy is False."""
        state = MagicMock()
        state.connection_healthy = False
        state.data_available = True
        state.last_data_fetch = time.time()

        assert get_staleness_severity(state) == "critical"

    def test_returns_fresh_when_no_data_yet(self) -> None:
        """Should return fresh when no data has been fetched."""
        state = MagicMock()
        state.connection_healthy = True
        state.data_available = False
        state.last_data_fetch = 0

        assert get_staleness_severity(state) == "fresh"

    def test_returns_fresh_for_recent_data(self) -> None:
        """Should return fresh when data is less than 10s old."""
        state = MagicMock()
        state.connection_healthy = True
        state.data_available = True
        state.last_data_fetch = time.time() - 5

        assert get_staleness_severity(state) == "fresh"

    def test_returns_stale_for_aging_data(self) -> None:
        """Should return stale when data is 10-30s old."""
        state = MagicMock()
        state.connection_healthy = True
        state.data_available = True
        state.last_data_fetch = time.time() - 20

        assert get_staleness_severity(state) == "stale"

    def test_returns_critical_for_old_data(self) -> None:
        """Should return critical when data is >30s old."""
        state = MagicMock()
        state.connection_healthy = True
        state.data_available = True
        state.last_data_fetch = time.time() - 45

        assert get_staleness_severity(state) == "critical"


class TestFormatFreshnessLabel:
    """Tests for format_freshness_label function."""

    def test_formats_seconds(self) -> None:
        """Should format as seconds for age < 60."""
        assert format_freshness_label(5) == "5s ago"
        assert format_freshness_label(45) == "45s ago"

    def test_formats_minutes(self) -> None:
        """Should format as minutes for age 60-3600."""
        assert format_freshness_label(120) == "2m ago"
        assert format_freshness_label(3000) == "50m ago"

    def test_formats_hours(self) -> None:
        """Should format as hours for age >= 3600."""
        assert format_freshness_label(3600) == "1h ago"
        assert format_freshness_label(7200) == "2h ago"


class TestGetFreshnessDisplay:
    """Tests for get_freshness_display function."""

    def test_returns_syncing_when_fetching(self) -> None:
        """Should return syncing when data_fetching is True."""
        state = MagicMock()
        state.data_fetching = True

        result = get_freshness_display(state)
        assert result is not None
        text, css_class = result
        assert "syncing" in text
        assert css_class == "fresh"

    def test_returns_none_when_no_data(self) -> None:
        """Should return None when no data has been fetched."""
        state = MagicMock()
        state.data_fetching = False
        state.data_available = False
        state.last_data_fetch = 0

        assert get_freshness_display(state) is None

    def test_returns_dim_for_fresh_data(self) -> None:
        """Should return dim styling for fresh data."""
        state = MagicMock()
        state.data_fetching = False
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 5

        result = get_freshness_display(state)
        assert result is not None
        text, css_class = result
        assert "[dim]" in text
        assert css_class == "fresh"

    def test_returns_yellow_for_stale_data(self) -> None:
        """Should return yellow styling for stale data."""
        state = MagicMock()
        state.data_fetching = False
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 20

        result = get_freshness_display(state)
        assert result is not None
        text, css_class = result
        assert "[yellow]" in text
        assert css_class == "stale"

    def test_returns_red_for_critical_data(self) -> None:
        """Should return red styling for critical data."""
        state = MagicMock()
        state.data_fetching = False
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 45

        result = get_freshness_display(state)
        assert result is not None
        text, css_class = result
        assert "[red]" in text
        assert css_class == "critical"


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

        assert get_staleness_banner(state) is None

    def test_returns_warning_when_stale(self) -> None:
        """Should return warning when data is stale."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 20

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
