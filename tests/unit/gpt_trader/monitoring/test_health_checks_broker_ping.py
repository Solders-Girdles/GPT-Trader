"""Tests for the broker ping health check."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from gpt_trader.monitoring.health_checks import check_broker_ping


class TestCheckBrokerPing:
    """Tests for check_broker_ping function."""

    def test_success_with_get_time(self) -> None:
        """Test successful ping using get_time method."""
        broker = MagicMock()
        broker.get_time.return_value = {"epoch": 1234567890}

        healthy, details = check_broker_ping(broker)

        assert healthy is True
        assert "latency_ms" in details
        assert details["method"] == "get_time"
        assert details["severity"] == "critical"
        broker.get_time.assert_called_once()

    def test_success_fallback_to_list_balances(self) -> None:
        """Test fallback to list_balances when get_time not available."""
        broker = MagicMock(spec=["list_balances"])
        broker.list_balances.return_value = [{"currency": "USD", "available": "100"}]

        healthy, details = check_broker_ping(broker)

        assert healthy is True
        assert details["method"] == "list_balances"
        broker.list_balances.assert_called_once()

    def test_failure_on_exception(self) -> None:
        """Test failure when broker call raises exception."""
        broker = MagicMock()
        broker.get_time.side_effect = ConnectionError("connection refused")

        healthy, details = check_broker_ping(broker)

        assert healthy is False
        assert "error" in details
        assert details["error_type"] == "ConnectionError"
        assert details["severity"] == "critical"

    def test_high_latency_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that high latency sets severity to warning."""
        broker = MagicMock()

        # Simulate slow response by mocking time
        mock_time = MagicMock()
        # First call returns 0, second returns 2.5 (2500ms latency)
        mock_time.side_effect = [0, 2.5]
        monkeypatch.setattr(time, "perf_counter", mock_time)

        healthy, details = check_broker_ping(broker)

        assert healthy is True
        assert details["latency_ms"] == 2500.0
        assert details["severity"] == "warning"
        assert "warning" in details
