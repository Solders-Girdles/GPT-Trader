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


class TestValidationAlerts:
    """Test suite for validation failure alert rules."""

    def test_validation_rules_registered(self, alert_manager):
        """Test that validation alert rules are registered."""
        status = alert_manager.get_rule_status()
        assert "validation_escalation" in status
        assert "validation_failures" in status

    def test_validation_escalation_alert_triggers(self, alert_manager, mock_app, sample_state):
        """Test validation escalation alert triggers when escalated."""
        sample_state.system_data.validation_escalated = True
        sample_state.system_data.validation_failures = {"mark_staleness": 5}

        alerts = alert_manager.check_alerts(sample_state)

        escalation_alerts = [a for a in alerts if a.rule_id == "validation_escalation"]
        assert len(escalation_alerts) == 1
        assert escalation_alerts[0].severity == AlertSeverity.ERROR
        assert escalation_alerts[0].category == AlertCategory.RISK
        assert "5" in escalation_alerts[0].message
        assert "reduce-only" in escalation_alerts[0].message.lower()

    def test_validation_escalation_no_alert_when_not_escalated(
        self, alert_manager, mock_app, sample_state
    ):
        """Test no validation escalation alert when not escalated."""
        sample_state.system_data.validation_escalated = False
        sample_state.system_data.validation_failures = {"mark_staleness": 2}

        alerts = alert_manager.check_alerts(sample_state)

        escalation_alerts = [a for a in alerts if a.rule_id == "validation_escalation"]
        assert len(escalation_alerts) == 0

    def test_validation_failures_alert_triggers_at_threshold(
        self, alert_manager, mock_app, sample_state
    ):
        """Test validation failures warning triggers at 2+ failures."""
        sample_state.system_data.validation_escalated = False
        sample_state.system_data.validation_failures = {"mark_staleness": 2, "slippage_guard": 1}

        alerts = alert_manager.check_alerts(sample_state)

        failure_alerts = [a for a in alerts if a.rule_id == "validation_failures"]
        assert len(failure_alerts) == 1
        assert failure_alerts[0].severity == AlertSeverity.WARNING
        assert failure_alerts[0].category == AlertCategory.SYSTEM
        assert "3 validation failures" in failure_alerts[0].message
        assert "mark_staleness" in failure_alerts[0].message

    def test_validation_failures_no_alert_below_threshold(
        self, alert_manager, mock_app, sample_state
    ):
        """Test no validation failures alert below 2 failures."""
        sample_state.system_data.validation_escalated = False
        sample_state.system_data.validation_failures = {"mark_staleness": 1}

        alerts = alert_manager.check_alerts(sample_state)

        failure_alerts = [a for a in alerts if a.rule_id == "validation_failures"]
        assert len(failure_alerts) == 0

    def test_validation_failures_no_alert_when_escalated(
        self, alert_manager, mock_app, sample_state
    ):
        """Test validation failures warning doesn't fire when already escalated."""
        sample_state.system_data.validation_escalated = True
        sample_state.system_data.validation_failures = {"mark_staleness": 5}

        alerts = alert_manager.check_alerts(sample_state)

        escalation_alerts = [a for a in alerts if a.rule_id == "validation_escalation"]
        failure_alerts = [a for a in alerts if a.rule_id == "validation_failures"]
        assert len(escalation_alerts) == 1
        assert len(failure_alerts) == 0

    def test_validation_alerts_handle_missing_data(self, alert_manager, mock_app, sample_state):
        """Test validation alerts gracefully handle missing data."""
        alerts = alert_manager.check_alerts(sample_state)

        validation_alerts = [
            a for a in alerts if a.rule_id in {"validation_escalation", "validation_failures"}
        ]
        assert len(validation_alerts) == 0
