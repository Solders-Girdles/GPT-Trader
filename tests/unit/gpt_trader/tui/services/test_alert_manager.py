from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tests.unit.gpt_trader.tui.services.alert_manager_test_utils import (  # naming: allow
    create_alert_manager,
    create_mock_app,
    create_sample_state,
)

from gpt_trader.tui.services.alert_manager import (
    AlertManager,
    AlertRule,
    AlertSeverity,
)


@pytest.fixture
def mock_app() -> MagicMock:
    """Create a mock TraderApp."""
    return create_mock_app()


@pytest.fixture
def alert_manager(mock_app: MagicMock) -> AlertManager:
    """Create an AlertManager with mock app."""
    return create_alert_manager(mock_app)


@pytest.fixture
def sample_state():
    """Create a test TuiState."""
    return create_sample_state()


class TestAlertManager:
    """Test suite for AlertManager."""

    def test_default_rules_registered(self, alert_manager):
        """Test that default rules are registered on init."""
        status = alert_manager.get_rule_status()
        assert "connection_lost" in status
        assert "rate_limit_high" in status
        assert "reduce_only_active" in status
        assert "daily_loss_warning" in status
        assert "large_unrealized_loss" in status
        assert "bot_stopped" in status

    def test_add_custom_rule(self, alert_manager):
        """Test adding a custom alert rule."""
        custom_rule = AlertRule(
            rule_id="custom_test",
            title="Custom Test",
            condition=lambda state: (True, "Test message"),
            severity=AlertSeverity.INFORMATION,
        )
        alert_manager.add_rule(custom_rule)

        status = alert_manager.get_rule_status()
        assert "custom_test" in status
        assert status["custom_test"]["title"] == "Custom Test"

    def test_remove_rule(self, alert_manager):
        """Test removing an alert rule."""
        result = alert_manager.remove_rule("connection_lost")
        assert result is True

        status = alert_manager.get_rule_status()
        assert "connection_lost" not in status

    def test_remove_nonexistent_rule(self, alert_manager):
        """Test removing a rule that doesn't exist."""
        result = alert_manager.remove_rule("nonexistent")
        assert result is False

    def test_enable_disable_rule(self, alert_manager):
        """Test enabling and disabling rules."""
        alert_manager.disable_rule("connection_lost")
        status = alert_manager.get_rule_status()
        assert status["connection_lost"]["enabled"] is False

        alert_manager.enable_rule("connection_lost")
        status = alert_manager.get_rule_status()
        assert status["connection_lost"]["enabled"] is True

    def test_alert_triggered_on_condition(self, alert_manager, mock_app, sample_state):
        """Test that alerts are triggered when conditions are met."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alerts = alert_manager.check_alerts(sample_state)

        assert len(alerts) == 1
        assert alerts[0].rule_id == "connection_lost"
        assert alerts[0].severity == AlertSeverity.ERROR
        mock_app.notify.assert_called_once()

    def test_alert_cooldown(self, alert_manager, mock_app, sample_state):
        """Test that cooldown prevents repeated alerts."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alerts1 = alert_manager.check_alerts(sample_state)
        assert len(alerts1) == 1

        alerts2 = alert_manager.check_alerts(sample_state)
        assert len(alerts2) == 0

        assert mock_app.notify.call_count == 1

    def test_no_alert_when_condition_not_met(self, alert_manager, mock_app, sample_state):
        """Test that no alerts trigger when conditions aren't met."""
        alerts = alert_manager.check_alerts(sample_state)

        assert len(alerts) == 0
        mock_app.notify.assert_not_called()

    def test_rate_limit_alert(self, alert_manager, mock_app, sample_state):
        """Test rate limit warning alert."""
        sample_state.system_data.rate_limit_usage = "85%"

        alerts = alert_manager.check_alerts(sample_state)

        rate_limit_alerts = [a for a in alerts if a.rule_id == "rate_limit_high"]
        assert len(rate_limit_alerts) == 1

    def test_reduce_only_alert(self, alert_manager, mock_app, sample_state):
        """Test reduce-only mode alert."""
        sample_state.risk_data.reduce_only_mode = True
        sample_state.risk_data.reduce_only_reason = "Daily loss limit reached"

        alerts = alert_manager.check_alerts(sample_state)

        reduce_only_alerts = [a for a in alerts if a.rule_id == "reduce_only_active"]
        assert len(reduce_only_alerts) == 1
        assert "Daily loss limit" in reduce_only_alerts[0].message

    def test_daily_loss_warning(self, alert_manager, mock_app, sample_state):
        """Test daily loss warning at 75% of limit."""
        sample_state.risk_data.daily_loss_limit_pct = 0.10  # 10%
        sample_state.risk_data.current_daily_loss_pct = 0.08  # 8% (80% of limit)

        alerts = alert_manager.check_alerts(sample_state)

        loss_alerts = [a for a in alerts if a.rule_id == "daily_loss_warning"]
        assert len(loss_alerts) == 1

    def test_bot_stopped_with_positions_alert(self, alert_manager, mock_app, sample_state):
        """Test alert when bot stops with open positions."""
        from decimal import Decimal

        from gpt_trader.tui.types import Position

        sample_state.running = False
        sample_state.position_data.positions = {
            "BTC-USD": Position(symbol="BTC-USD", quantity=Decimal("1.0"))
        }

        alerts = alert_manager.check_alerts(sample_state)

        stopped_alerts = [a for a in alerts if a.rule_id == "bot_stopped"]
        assert len(stopped_alerts) == 1
        assert "1 open position" in stopped_alerts[0].message

    def test_alert_history(self, alert_manager, sample_state):
        """Test alert history tracking."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alert_manager.check_alerts(sample_state)

        history = alert_manager.get_history()
        assert len(history) == 1
        assert history[0].rule_id == "connection_lost"

    def test_clear_history(self, alert_manager, sample_state):
        """Test clearing alert history."""
        sample_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(sample_state)

        alert_manager.clear_history()

        history = alert_manager.get_history()
        assert len(history) == 0

    def test_reset_cooldowns(self, alert_manager, mock_app, sample_state):
        """Test resetting all cooldowns."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alert_manager.check_alerts(sample_state)
        assert mock_app.notify.call_count == 1

        alert_manager.reset_cooldowns()

        alert_manager.check_alerts(sample_state)
        assert mock_app.notify.call_count == 2

    def test_disabled_rule_not_checked(self, alert_manager, mock_app, sample_state):
        """Test that disabled rules are not checked."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alert_manager.disable_rule("connection_lost")
        alerts = alert_manager.check_alerts(sample_state)

        connection_alerts = [a for a in alerts if a.rule_id == "connection_lost"]
        assert len(connection_alerts) == 0
