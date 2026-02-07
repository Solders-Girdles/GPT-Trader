"""Tests for health check helper functions."""

from __future__ import annotations

import pytest

from gpt_trader.app.health_server import (
    HealthState,
    add_health_check,
    mark_live,
    mark_ready,
)
from gpt_trader.monitoring.interfaces import HealthCheckResult


class TestHealthCheckFunctions:
    """Tests for health check helper functions."""

    @pytest.fixture
    def health_state(self) -> HealthState:
        """Fresh health state per test."""
        return HealthState()

    def test_mark_ready(self, health_state: HealthState) -> None:
        mark_ready(health_state, True, "all_systems_go")
        assert health_state.ready is True
        assert health_state.reason == "all_systems_go"

    def test_mark_live(self, health_state: HealthState) -> None:
        mark_live(health_state, False, "graceful_shutdown")
        assert health_state.live is False
        assert health_state.reason == "graceful_shutdown"

    def test_add_health_check_success(self, health_state: HealthState) -> None:
        def check_broker() -> HealthCheckResult:
            return HealthCheckResult(healthy=True, details={"latency_ms": 25})

        add_health_check(health_state, "broker", check_broker)
        assert health_state.checks["broker"].status == "pass"
        assert health_state.checks["broker"].details["latency_ms"] == 25
        assert health_state.checks_payload()["broker"]["status"] == "pass"

    def test_add_health_check_failure(self, health_state: HealthState) -> None:
        def check_database() -> HealthCheckResult:
            return HealthCheckResult(healthy=False, details={"error": "connection refused"})

        add_health_check(health_state, "database", check_database)
        assert health_state.checks["database"].status == "fail"

    def test_add_health_check_exception(self, health_state: HealthState) -> None:
        def check_flaky() -> HealthCheckResult:
            raise RuntimeError("check failed")

        add_health_check(health_state, "flaky", check_flaky)
        assert health_state.checks["flaky"].status == "fail"
        assert "check failed" in health_state.checks["flaky"].details["error"]
