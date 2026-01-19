"""Edge case tests for staleness helpers with partial/missing data."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from gpt_trader.tui.staleness_helpers import (
    get_connection_banner,
    get_data_age_seconds,
    get_empty_state_config,
    get_freshness_display,
    get_staleness_banner,
    get_staleness_severity,
)


class TestStalenessEdgeCases:
    """Edge case tests for staleness helpers with partial/missing data."""

    def test_get_data_age_with_negative_last_fetch(self) -> None:
        """Test that negative last_data_fetch is treated as no data."""
        state = MagicMock()
        state.data_available = True
        state.last_data_fetch = -1

        assert get_data_age_seconds(state) is None

    def test_connection_unhealthy_overrides_to_critical(self) -> None:
        """Test that connection_healthy=False always returns critical severity."""
        state = MagicMock()
        state.connection_healthy = False
        # Even with very fresh data
        state.data_available = True
        state.last_data_fetch = time.time() - 1  # 1 second ago

        assert get_staleness_severity(state) == "critical"

    def test_staleness_banner_with_connection_unhealthy_and_fresh_data(self) -> None:
        """Test banner includes reconnect hint when connection unhealthy."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        state.data_available = True
        state.connection_healthy = False  # Unhealthy
        state.last_data_fetch = time.time() - 5  # Fresh data
        # No execution issues
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 0

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "stale" in text.lower()
        assert "R" in text  # Reconnect hint
        assert severity == "error"

    def test_freshness_display_with_connection_unhealthy(self) -> None:
        """Test freshness display returns critical when connection unhealthy."""
        state = MagicMock()
        state.data_fetching = False
        state.data_available = True
        state.connection_healthy = False  # Unhealthy
        state.last_data_fetch = time.time() - 5  # Fresh data normally

        result = get_freshness_display(state)
        assert result is not None
        text, css_class = result
        assert "[red]" in text
        assert css_class == "critical"

    def test_staleness_banner_with_none_degraded_reason(self) -> None:
        """Test banner handles None degraded_reason gracefully."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = True
        state.degraded_reason = None  # No reason provided

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "Degraded" in text
        assert "Status reporter unavailable" in text  # Default message
        assert severity == "warning"

    def test_get_empty_state_config_handles_none_connection_status(self) -> None:
        """Test empty state config handles None connection status."""
        config = get_empty_state_config(
            data_type="Market",
            bot_running=True,
            data_source_mode="live",
            connection_status=None,  # type: ignore
        )
        # Should treat as transitional/connecting state
        assert "Connecting" in config["title"]

    def test_get_connection_banner_case_insensitive(self) -> None:
        """Test connection banner handles lowercase status."""
        result = get_connection_banner(
            connection_status="disconnected",  # lowercase
            bot_running=True,
            degraded_mode=False,
        )
        assert result is not None
        text, severity = result
        assert "Connection lost" in text
        assert severity == "error"

    def test_staleness_banner_returns_none_no_data_yet(self) -> None:
        """Test banner returns None when no data has been fetched yet."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        state.data_available = False  # No data yet
        state.connection_healthy = True
        state.last_data_fetch = 0
        # No execution data
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 0

        result = get_staleness_banner(state)
        assert result is None  # No banner for initial state

    def test_get_data_age_at_boundary_threshold(self) -> None:
        """Test staleness exactly at threshold boundaries."""
        state = MagicMock()
        state.connection_healthy = True
        state.data_available = True

        # Exactly at fresh threshold (10s)
        state.last_data_fetch = time.time() - 10
        assert get_staleness_severity(state) == "stale"

        # Exactly at stale threshold (30s)
        state.last_data_fetch = time.time() - 30
        assert get_staleness_severity(state) == "critical"
