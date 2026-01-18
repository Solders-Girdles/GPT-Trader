"""Tests for staleness freshness label + display helpers."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from gpt_trader.tui.staleness_helpers import (
    format_freshness_label,
    get_freshness_display,
)


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
