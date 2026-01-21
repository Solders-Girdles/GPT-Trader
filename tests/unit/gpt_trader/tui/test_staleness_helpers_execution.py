"""Tests for execution health banners and staleness integration."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from gpt_trader.tui.staleness_helpers import (
    get_execution_health_banner,
    get_staleness_banner,
)


class TestGetExecutionHealthBanner:
    """Tests for get_execution_health_banner function."""

    def test_returns_none_when_healthy(self) -> None:
        """Should return None when execution is healthy."""
        state = MagicMock()
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 100
        state.execution_data.success_rate = 98.0
        state.execution_data.retry_rate = 0.2

        assert get_execution_health_banner(state) is None

    def test_returns_none_when_no_executions(self) -> None:
        """Should return None when no executions have occurred."""
        state = MagicMock()
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 0

        assert get_execution_health_banner(state) is None

    def test_returns_error_when_circuit_open(self) -> None:
        """Should return error banner when circuit breaker is open."""
        state = MagicMock()
        state.resilience_data.any_circuit_open = True

        result = get_execution_health_banner(state)
        assert result is not None
        text, severity = result
        assert "Circuit breaker OPEN" in text
        assert severity == "error"

    def test_returns_error_when_success_rate_critical(self) -> None:
        """Should return error banner when success rate below critical threshold."""
        state = MagicMock()
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 100
        state.execution_data.success_rate = 75.0  # Below 80%

        result = get_execution_health_banner(state)
        assert result is not None
        text, severity = result
        assert "degraded" in text.lower()
        assert "75%" in text
        assert severity == "error"

    def test_returns_warning_when_success_rate_low(self) -> None:
        """Should return warning banner when success rate below warning threshold."""
        state = MagicMock()
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 100
        state.execution_data.success_rate = 90.0  # Below 95%, above 80%
        state.execution_data.retry_rate = 0.2

        result = get_execution_health_banner(state)
        assert result is not None
        text, severity = result
        assert "warning" in text.lower()
        assert "90%" in text
        assert severity == "warning"

    def test_returns_warning_when_retry_rate_high(self) -> None:
        """Should return warning banner when retry rate exceeds threshold."""
        state = MagicMock()
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 100
        state.execution_data.success_rate = 98.0  # Healthy
        state.execution_data.retry_rate = 0.8  # Above 0.5

        result = get_execution_health_banner(state)
        assert result is not None
        text, severity = result
        assert "retry rate" in text.lower()
        assert severity == "warning"

    def test_handles_missing_execution_data(self) -> None:
        """Should return None when execution_data is not available."""
        state = MagicMock()
        state.resilience_data.any_circuit_open = False
        # Simulate missing attribute
        del state.execution_data

        assert get_execution_health_banner(state) is None

    def test_handles_missing_resilience_data(self) -> None:
        """Should return None when resilience_data is not available."""
        state = MagicMock()
        # Simulate missing attribute
        del state.resilience_data

        assert get_execution_health_banner(state) is None


class TestExecutionHealthBannerIntegration:
    """Tests for execution health integration in get_staleness_banner."""

    def test_execution_health_takes_priority_over_staleness(self) -> None:
        """Execution health issues should show before data staleness warnings."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        # Data is stale (would normally show warning)
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 20  # Stale
        # But execution is degraded (takes priority)
        state.resilience_data.any_circuit_open = True

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "Circuit breaker" in text
        assert severity == "error"

    def test_degraded_mode_takes_priority_over_execution(self) -> None:
        """Degraded mode should take priority over execution health."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = True
        state.degraded_reason = "Test degraded"
        # Execution also degraded
        state.resilience_data.any_circuit_open = True

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "Degraded" in text
        assert "Test degraded" in text
        assert severity == "warning"

    def test_execution_critical_shows_over_stale_warning(self) -> None:
        """Critical execution should show instead of stale warning."""
        state = MagicMock()
        state.data_fetching = False
        state.degraded_mode = False
        # Data is stale
        state.data_available = True
        state.connection_healthy = True
        state.last_data_fetch = time.time() - 15  # Stale (10-30s)
        # Execution has critical success rate
        state.resilience_data.any_circuit_open = False
        state.execution_data.submissions_total = 100
        state.execution_data.success_rate = 70.0  # Critical

        result = get_staleness_banner(state)
        assert result is not None
        text, severity = result
        assert "degraded" in text.lower()
        assert severity == "error"
