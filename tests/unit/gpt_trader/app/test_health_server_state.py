"""Tests for health server state."""

from gpt_trader.app.health_server import DEFAULT_HEALTH_PORT, HealthState
from gpt_trader.monitoring.interfaces import HealthCheckResult


class TestHealthState:
    """Tests for HealthState dataclass."""

    def test_initial_state(self) -> None:
        state = HealthState()
        assert state.ready is False
        assert state.live is True
        assert state.reason == "initializing"
        assert state.checks == {}

    def test_set_ready(self) -> None:
        state = HealthState()
        state.set_ready(True, "test_ready")
        assert state.ready is True
        assert state.reason == "test_ready"

    def test_set_live(self) -> None:
        state = HealthState()
        state.set_live(False, "test_shutdown")
        assert state.live is False
        assert state.reason == "test_shutdown"

    def test_add_check_pass(self) -> None:
        state = HealthState()
        state.add_check(
            "broker",
            HealthCheckResult(healthy=True, details={"latency_ms": 50}),
        )
        assert state.checks["broker"].healthy is True
        assert state.checks["broker"].details["latency_ms"] == 50
        assert state.checks_payload()["broker"]["status"] == "pass"

    def test_add_check_fail(self) -> None:
        state = HealthState()
        state.add_check(
            "database",
            HealthCheckResult(healthy=False, details={"error": "connection timeout"}),
        )
        assert state.checks["database"].healthy is False
        assert state.checks["database"].details["error"] == "connection timeout"
        assert state.checks_payload()["database"]["status"] == "fail"


def test_default_port_is_8080() -> None:
    assert DEFAULT_HEALTH_PORT == 8080
