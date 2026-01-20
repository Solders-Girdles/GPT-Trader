from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.tui.services.alert_manager_test_utils import (  # naming: allow
    create_alert_manager,
    create_mock_app,
    create_sample_state,
)

from gpt_trader.tui.services.alert_manager import (
    AlertCategory,
    AlertManager,
    AlertSeverity,
)


@pytest.fixture
def mock_app() -> MagicMock:
    return create_mock_app()


@pytest.fixture
def alert_manager(mock_app: MagicMock) -> AlertManager:
    return create_alert_manager(mock_app)


@pytest.fixture
def sample_state():
    return create_sample_state()


class TestExecutionHealthAlerts:
    """Test suite for execution health alert rules."""

    def test_execution_rules_registered(self, alert_manager):
        """Test that execution health rules are registered on init."""
        status = alert_manager.get_rule_status()
        assert "circuit_breaker_open" in status
        assert "execution_critical" in status
        assert "execution_degraded" in status
        assert "execution_p95_spike" in status
        assert "execution_retry_high" in status

    def test_circuit_breaker_alert(self, alert_manager, mock_app, sample_state):
        """Test circuit breaker open alert."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = True

        alerts = alert_manager.check_alerts(sample_state)

        cb_alerts = [a for a in alerts if a.rule_id == "circuit_breaker_open"]
        assert len(cb_alerts) == 1
        assert cb_alerts[0].severity == AlertSeverity.ERROR
        assert "paused" in cb_alerts[0].message.lower()

    def test_circuit_breaker_no_alert_when_closed(self, alert_manager, mock_app, sample_state):
        """Test no alert when circuit breaker is closed."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False

        alerts = alert_manager.check_alerts(sample_state)

        cb_alerts = [a for a in alerts if a.rule_id == "circuit_breaker_open"]
        assert len(cb_alerts) == 0

    def test_execution_critical_alert(self, alert_manager, mock_app, sample_state):
        """Test execution critical alert when success rate drops below 80%."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 10
        sample_state.execution_data.success_rate = 70.0  # Below 80%

        alerts = alert_manager.check_alerts(sample_state)

        critical_alerts = [a for a in alerts if a.rule_id == "execution_critical"]
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.ERROR
        assert "70%" in critical_alerts[0].message

    def test_execution_critical_requires_sample_size(self, alert_manager, mock_app, sample_state):
        """Test execution critical alert requires minimum sample size."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 3  # Below threshold of 5
        sample_state.execution_data.success_rate = 50.0

        alerts = alert_manager.check_alerts(sample_state)

        critical_alerts = [a for a in alerts if a.rule_id == "execution_critical"]
        assert len(critical_alerts) == 0

    def test_execution_degraded_alert(self, alert_manager, mock_app, sample_state):
        """Test execution degraded alert when success rate is 80-95%."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 20
        sample_state.execution_data.success_rate = 90.0  # Between 80% and 95%

        alerts = alert_manager.check_alerts(sample_state)

        degraded_alerts = [a for a in alerts if a.rule_id == "execution_degraded"]
        assert len(degraded_alerts) == 1
        assert degraded_alerts[0].severity == AlertSeverity.WARNING
        assert "90%" in degraded_alerts[0].message

    def test_execution_degraded_not_triggered_when_critical(
        self, alert_manager, mock_app, sample_state
    ):
        """Test degraded alert doesn't fire when already critical (<80%)."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 20
        sample_state.execution_data.success_rate = 70.0  # Below 80%

        alerts = alert_manager.check_alerts(sample_state)

        critical_alerts = [a for a in alerts if a.rule_id == "execution_critical"]
        degraded_alerts = [a for a in alerts if a.rule_id == "execution_degraded"]
        assert len(critical_alerts) == 1
        assert len(degraded_alerts) == 0

    def test_execution_degraded_requires_sample_size(self, alert_manager, mock_app, sample_state):
        """Test execution degraded alert requires minimum sample size."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 5  # Below threshold of 10
        sample_state.execution_data.success_rate = 90.0

        alerts = alert_manager.check_alerts(sample_state)

        degraded_alerts = [a for a in alerts if a.rule_id == "execution_degraded"]
        assert len(degraded_alerts) == 0

    def test_p95_latency_alert(self, alert_manager, mock_app, sample_state):
        """Test p95 latency spike alert."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 20
        sample_state.execution_data.success_rate = 98.0  # Healthy
        sample_state.execution_data.p95_latency_ms = 750.0  # Above 500ms

        alerts = alert_manager.check_alerts(sample_state)

        latency_alerts = [a for a in alerts if a.rule_id == "execution_p95_spike"]
        assert len(latency_alerts) == 1
        assert latency_alerts[0].severity == AlertSeverity.WARNING
        assert "750" in latency_alerts[0].message

    def test_p95_latency_no_alert_when_normal(self, alert_manager, mock_app, sample_state):
        """Test no p95 latency alert when latency is normal."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 20
        sample_state.execution_data.success_rate = 98.0
        sample_state.execution_data.p95_latency_ms = 200.0  # Normal

        alerts = alert_manager.check_alerts(sample_state)

        latency_alerts = [a for a in alerts if a.rule_id == "execution_p95_spike"]
        assert len(latency_alerts) == 0

    def test_retry_rate_alert(self, alert_manager, mock_app, sample_state):
        """Test high retry rate alert."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 20
        sample_state.execution_data.success_rate = 98.0  # Healthy
        sample_state.execution_data.p95_latency_ms = 200.0  # Normal
        sample_state.execution_data.retry_rate = 0.8  # Above 0.5

        alerts = alert_manager.check_alerts(sample_state)

        retry_alerts = [a for a in alerts if a.rule_id == "execution_retry_high"]
        assert len(retry_alerts) == 1
        assert retry_alerts[0].severity == AlertSeverity.WARNING
        assert "0.8" in retry_alerts[0].message

    def test_retry_rate_no_alert_when_normal(self, alert_manager, mock_app, sample_state):
        """Test no retry rate alert when rate is normal."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = False
        sample_state.execution_data = MagicMock()
        sample_state.execution_data.submissions_total = 20
        sample_state.execution_data.success_rate = 98.0
        sample_state.execution_data.p95_latency_ms = 200.0
        sample_state.execution_data.retry_rate = 0.2  # Normal

        alerts = alert_manager.check_alerts(sample_state)

        retry_alerts = [a for a in alerts if a.rule_id == "execution_retry_high"]
        assert len(retry_alerts) == 0

    def test_execution_alerts_handle_missing_data(self, alert_manager, mock_app, sample_state):
        """Test execution alerts gracefully handle missing data attributes."""
        alerts = alert_manager.check_alerts(sample_state)

        exec_alerts = [
            a
            for a in alerts
            if a.rule_id
            in {
                "circuit_breaker_open",
                "execution_critical",
                "execution_degraded",
                "execution_p95_spike",
                "execution_retry_high",
            }
        ]
        assert len(exec_alerts) == 0

    def test_execution_alerts_have_system_category(self, alert_manager, mock_app, sample_state):
        """Test that execution alerts have SYSTEM category."""
        sample_state.resilience_data = MagicMock()
        sample_state.resilience_data.any_circuit_open = True

        alerts = alert_manager.check_alerts(sample_state)

        cb_alerts = [a for a in alerts if a.rule_id == "circuit_breaker_open"]
        assert len(cb_alerts) == 1
        assert cb_alerts[0].category == AlertCategory.SYSTEM
