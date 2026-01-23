"""Edge-case tests for EnvironmentMonitor."""

from __future__ import annotations

from gpt_trader.monitoring.configuration_guardian.environment import EnvironmentMonitor
from gpt_trader.monitoring.configuration_guardian.models import BaselineSnapshot
from gpt_trader.monitoring.configuration_guardian.responses import DriftResponse


def test_critical_env_change_emits_shutdown_event(monkeypatch) -> None:
    var_name = "COINBASE_ENABLE_INTX_PERPS"
    monkeypatch.setenv(var_name, "0")
    monitor = EnvironmentMonitor(BaselineSnapshot())

    monkeypatch.setenv(var_name, "1")
    events = monitor.check_changes()

    assert len(events) == 1
    event = events[0]
    assert event.drift_type == "critical_env_changed"
    assert event.severity == "critical"
    assert event.suggested_response == DriftResponse.EMERGENCY_SHUTDOWN
    assert event.applied_response == DriftResponse.EMERGENCY_SHUTDOWN


def test_risk_env_change_emits_reduce_only_event(monkeypatch) -> None:
    var_name = "PERPS_POSITION_FRACTION"
    monkeypatch.setenv(var_name, "0.1")
    monitor = EnvironmentMonitor(BaselineSnapshot())

    monkeypatch.setenv(var_name, "0.2")
    events = monitor.check_changes()

    assert len(events) == 1
    event = events[0]
    assert event.drift_type == "risk_env_changed"
    assert event.severity == "high"
    assert event.suggested_response == DriftResponse.REDUCE_ONLY
    assert event.applied_response == DriftResponse.REDUCE_ONLY


def test_monitor_env_change_emits_sticky_event(monkeypatch) -> None:
    var_name = "COINBASE_DEFAULT_QUOTE"
    monkeypatch.setenv(var_name, "USD")
    monitor = EnvironmentMonitor(BaselineSnapshot())

    monkeypatch.setenv(var_name, "USDC")
    events = monitor.check_changes()

    assert len(events) == 1
    event = events[0]
    assert event.drift_type == "monitored_env_changed"
    assert event.severity == "low"
    assert event.suggested_response == DriftResponse.STICKY
    assert event.applied_response == DriftResponse.STICKY


def test_last_state_updates_after_check(monkeypatch) -> None:
    var_name = "MOCK_BROKER"
    monkeypatch.setenv(var_name, "0")
    monitor = EnvironmentMonitor(BaselineSnapshot())

    monkeypatch.setenv(var_name, "1")
    assert monitor.check_changes()
    assert monitor.check_changes() == []


def test_capture_state_redacts_sensitive_env_vars(monkeypatch) -> None:
    monkeypatch.setattr(EnvironmentMonitor, "CRITICAL_ENV_VARS", {"TEST_API_KEY"})
    monkeypatch.setenv("TEST_API_KEY", "secret-value")

    monitor = EnvironmentMonitor(BaselineSnapshot())
    state = monitor.get_current_state()

    assert state["TEST_API_KEY"] == "[REDACTED]"
