"""Tests for AlertManager service."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.services.alert_manager import (
    Alert,
    AlertCategory,
    AlertManager,
    AlertRule,
    AlertSeverity,
)
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.types import PortfolioSummary, RiskState, SystemStatus


@pytest.fixture
def mock_app():
    """Create a mock TraderApp."""
    app = MagicMock()
    app.notify = MagicMock()
    return app


@pytest.fixture
def alert_manager(mock_app):
    """Create an AlertManager with mock app."""
    return AlertManager(mock_app)


@pytest.fixture
def test_state():
    """Create a test TuiState."""
    state = TuiState()
    state.system_data = SystemStatus(
        connection_status="CONNECTED",
        rate_limit_usage="50%",
    )
    state.risk_data = RiskState(
        reduce_only_mode=False,
        daily_loss_limit_pct=0.05,
        current_daily_loss_pct=0.0,
    )
    state.position_data = PortfolioSummary(
        positions={},
        total_unrealized_pnl=Decimal("0"),
    )
    state.running = True
    return state


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

    def test_alert_triggered_on_condition(self, alert_manager, mock_app, test_state):
        """Test that alerts are triggered when conditions are met."""
        # Set up a condition that will trigger
        test_state.system_data.connection_status = "DISCONNECTED"

        alerts = alert_manager.check_alerts(test_state)

        assert len(alerts) == 1
        assert alerts[0].rule_id == "connection_lost"
        assert alerts[0].severity == AlertSeverity.ERROR
        mock_app.notify.assert_called_once()

    def test_alert_cooldown(self, alert_manager, mock_app, test_state):
        """Test that cooldown prevents repeated alerts."""
        test_state.system_data.connection_status = "DISCONNECTED"

        # First alert should trigger
        alerts1 = alert_manager.check_alerts(test_state)
        assert len(alerts1) == 1

        # Second check should be blocked by cooldown
        alerts2 = alert_manager.check_alerts(test_state)
        assert len(alerts2) == 0

        # Notify should only be called once
        assert mock_app.notify.call_count == 1

    def test_no_alert_when_condition_not_met(self, alert_manager, mock_app, test_state):
        """Test that no alerts trigger when conditions aren't met."""
        # test_state has all good values by default
        alerts = alert_manager.check_alerts(test_state)

        # No alerts should trigger (state is healthy)
        assert len(alerts) == 0
        mock_app.notify.assert_not_called()

    def test_rate_limit_alert(self, alert_manager, mock_app, test_state):
        """Test rate limit warning alert."""
        test_state.system_data.rate_limit_usage = "85%"

        alerts = alert_manager.check_alerts(test_state)

        # Should trigger rate limit warning
        rate_limit_alerts = [a for a in alerts if a.rule_id == "rate_limit_high"]
        assert len(rate_limit_alerts) == 1

    def test_reduce_only_alert(self, alert_manager, mock_app, test_state):
        """Test reduce-only mode alert."""
        test_state.risk_data.reduce_only_mode = True
        test_state.risk_data.reduce_only_reason = "Daily loss limit reached"

        alerts = alert_manager.check_alerts(test_state)

        reduce_only_alerts = [a for a in alerts if a.rule_id == "reduce_only_active"]
        assert len(reduce_only_alerts) == 1
        assert "Daily loss limit" in reduce_only_alerts[0].message

    def test_daily_loss_warning(self, alert_manager, mock_app, test_state):
        """Test daily loss warning at 75% of limit."""
        test_state.risk_data.daily_loss_limit_pct = 0.10  # 10%
        test_state.risk_data.current_daily_loss_pct = 0.08  # 8% (80% of limit)

        alerts = alert_manager.check_alerts(test_state)

        loss_alerts = [a for a in alerts if a.rule_id == "daily_loss_warning"]
        assert len(loss_alerts) == 1

    def test_bot_stopped_with_positions_alert(self, alert_manager, mock_app, test_state):
        """Test alert when bot stops with open positions."""
        from gpt_trader.tui.types import Position

        test_state.running = False
        test_state.position_data.positions = {
            "BTC-USD": Position(symbol="BTC-USD", quantity=Decimal("1.0"))
        }

        alerts = alert_manager.check_alerts(test_state)

        stopped_alerts = [a for a in alerts if a.rule_id == "bot_stopped"]
        assert len(stopped_alerts) == 1
        assert "1 open position" in stopped_alerts[0].message

    def test_alert_history(self, alert_manager, test_state):
        """Test alert history tracking."""
        test_state.system_data.connection_status = "DISCONNECTED"

        alert_manager.check_alerts(test_state)

        history = alert_manager.get_history()
        assert len(history) == 1
        assert history[0].rule_id == "connection_lost"

    def test_clear_history(self, alert_manager, test_state):
        """Test clearing alert history."""
        test_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(test_state)

        alert_manager.clear_history()

        history = alert_manager.get_history()
        assert len(history) == 0

    def test_reset_cooldowns(self, alert_manager, mock_app, test_state):
        """Test resetting all cooldowns."""
        test_state.system_data.connection_status = "DISCONNECTED"

        # First alert
        alert_manager.check_alerts(test_state)
        assert mock_app.notify.call_count == 1

        # Reset cooldowns
        alert_manager.reset_cooldowns()

        # Should trigger again after reset
        alert_manager.check_alerts(test_state)
        assert mock_app.notify.call_count == 2

    def test_disabled_rule_not_checked(self, alert_manager, mock_app, test_state):
        """Test that disabled rules are not checked."""
        test_state.system_data.connection_status = "DISCONNECTED"

        alert_manager.disable_rule("connection_lost")
        alerts = alert_manager.check_alerts(test_state)

        connection_alerts = [a for a in alerts if a.rule_id == "connection_lost"]
        assert len(connection_alerts) == 0


class TestAlertCategories:
    """Test suite for AlertCategory functionality."""

    def test_alert_category_values(self):
        """Test that AlertCategory has expected values."""
        assert AlertCategory.TRADE.value == "trade"
        assert AlertCategory.POSITION.value == "position"
        assert AlertCategory.STRATEGY.value == "strategy"
        assert AlertCategory.RISK.value == "risk"
        assert AlertCategory.SYSTEM.value == "system"
        assert AlertCategory.ERROR.value == "error"

    def test_alert_has_category(self, alert_manager, test_state):
        """Test that triggered alerts have a category."""
        test_state.system_data.connection_status = "DISCONNECTED"

        alerts = alert_manager.check_alerts(test_state)

        assert len(alerts) == 1
        assert alerts[0].category == AlertCategory.SYSTEM

    def test_default_rules_have_categories(self, alert_manager):
        """Test that all default rules have appropriate categories."""
        # Check rule categories through status
        status = alert_manager.get_rule_status()

        # System category rules
        assert "connection_lost" in status
        assert "rate_limit_high" in status
        assert "bot_stopped" in status

        # Risk category rules
        assert "reduce_only_active" in status
        assert "daily_loss_warning" in status

        # Position category rules
        assert "large_unrealized_loss" in status

    def test_filter_history_by_category(self, alert_manager, test_state):
        """Test filtering alert history by category."""
        # Trigger a SYSTEM category alert
        test_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(test_state)

        # Get all history
        all_history = alert_manager.get_history()
        assert len(all_history) == 1

        # Filter by SYSTEM category
        system_history = alert_manager.get_history(categories={AlertCategory.SYSTEM})
        assert len(system_history) == 1

        # Filter by TRADE category (should be empty)
        trade_history = alert_manager.get_history(categories={AlertCategory.TRADE})
        assert len(trade_history) == 0

    def test_filter_history_by_severity(self, alert_manager, test_state):
        """Test filtering alert history by minimum severity."""
        # Trigger an ERROR severity alert
        test_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(test_state)

        # Get all history (should include ERROR)
        all_history = alert_manager.get_history()
        assert len(all_history) == 1
        assert all_history[0].severity == AlertSeverity.ERROR

        # Filter by ERROR severity (should include)
        error_history = alert_manager.get_history(min_severity=AlertSeverity.ERROR)
        assert len(error_history) == 1

        # Filter by WARNING severity (should include ERROR since it's higher)
        warning_history = alert_manager.get_history(min_severity=AlertSeverity.WARNING)
        assert len(warning_history) == 1

    def test_get_history_by_category(self, alert_manager, test_state):
        """Test get_history_by_category convenience method."""
        test_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(test_state)

        system_alerts = alert_manager.get_history_by_category(AlertCategory.SYSTEM)
        assert len(system_alerts) == 1

        trade_alerts = alert_manager.get_history_by_category(AlertCategory.TRADE)
        assert len(trade_alerts) == 0

    def test_get_category_counts(self, alert_manager, test_state):
        """Test get_category_counts method."""
        # Initially all counts should be 0
        counts = alert_manager.get_category_counts()
        assert all(count == 0 for count in counts.values())

        # Trigger a SYSTEM alert
        test_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(test_state)

        counts = alert_manager.get_category_counts()
        assert counts[AlertCategory.SYSTEM] == 1
        assert counts[AlertCategory.TRADE] == 0

    def test_custom_rule_with_category(self, alert_manager):
        """Test adding a custom rule with a specific category."""
        custom_rule = AlertRule(
            rule_id="custom_trade",
            title="Custom Trade Alert",
            condition=lambda state: (True, "Trade executed"),
            severity=AlertSeverity.INFORMATION,
            category=AlertCategory.TRADE,
        )
        alert_manager.add_rule(custom_rule)

        status = alert_manager.get_rule_status()
        assert "custom_trade" in status

    def test_risk_alert_category(self, alert_manager, test_state):
        """Test that risk alerts have RISK category."""
        test_state.risk_data.reduce_only_mode = True
        test_state.risk_data.reduce_only_reason = "Test reason"

        alerts = alert_manager.check_alerts(test_state)

        risk_alerts = [a for a in alerts if a.category == AlertCategory.RISK]
        assert len(risk_alerts) == 1

    def test_position_alert_category(self, alert_manager, test_state):
        """Test that position alerts have POSITION category."""
        test_state.position_data.total_unrealized_pnl = Decimal("-600")

        alerts = alert_manager.check_alerts(test_state)

        position_alerts = [a for a in alerts if a.category == AlertCategory.POSITION]
        assert len(position_alerts) == 1
