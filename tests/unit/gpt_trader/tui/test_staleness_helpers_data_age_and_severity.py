"""Tests for staleness helper age/severity calculations."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from gpt_trader.tui.staleness_helpers import (
    get_data_age_seconds,
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
