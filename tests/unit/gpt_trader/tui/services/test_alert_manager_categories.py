from __future__ import annotations

from decimal import Decimal
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
    AlertRule,
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

    def test_alert_has_category(self, alert_manager, sample_state):
        """Test that triggered alerts have a category."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alerts = alert_manager.check_alerts(sample_state)

        assert len(alerts) == 1
        assert alerts[0].category == AlertCategory.SYSTEM

    def test_default_rules_have_categories(self, alert_manager):
        """Test that all default rules have appropriate categories."""
        status = alert_manager.get_rule_status()

        assert "connection_lost" in status
        assert "rate_limit_high" in status
        assert "bot_stopped" in status

        assert "reduce_only_active" in status
        assert "daily_loss_warning" in status

        assert "large_unrealized_loss" in status

    def test_filter_history_by_category(self, alert_manager, sample_state):
        """Test filtering alert history by category."""
        sample_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(sample_state)

        all_history = alert_manager.get_history()
        assert len(all_history) == 1

        system_history = alert_manager.get_history(categories={AlertCategory.SYSTEM})
        assert len(system_history) == 1

        trade_history = alert_manager.get_history(categories={AlertCategory.TRADE})
        assert len(trade_history) == 0

    def test_filter_history_by_severity(self, alert_manager, sample_state):
        """Test filtering alert history by minimum severity."""
        sample_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(sample_state)

        all_history = alert_manager.get_history()
        assert len(all_history) == 1
        assert all_history[0].severity == AlertSeverity.ERROR

        error_history = alert_manager.get_history(min_severity=AlertSeverity.ERROR)
        assert len(error_history) == 1

        warning_history = alert_manager.get_history(min_severity=AlertSeverity.WARNING)
        assert len(warning_history) == 1

    def test_get_history_by_category(self, alert_manager, sample_state):
        """Test get_history_by_category convenience method."""
        sample_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(sample_state)

        system_alerts = alert_manager.get_history_by_category(AlertCategory.SYSTEM)
        assert len(system_alerts) == 1

        trade_alerts = alert_manager.get_history_by_category(AlertCategory.TRADE)
        assert len(trade_alerts) == 0

    def test_get_category_counts(self, alert_manager, sample_state):
        """Test get_category_counts method."""
        counts = alert_manager.get_category_counts()
        assert all(count == 0 for count in counts.values())

        sample_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(sample_state)

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

    def test_risk_alert_category(self, alert_manager, sample_state):
        """Test that risk alerts have RISK category."""
        sample_state.risk_data.reduce_only_mode = True
        sample_state.risk_data.reduce_only_reason = "Test reason"

        alerts = alert_manager.check_alerts(sample_state)

        risk_alerts = [a for a in alerts if a.category == AlertCategory.RISK]
        assert len(risk_alerts) == 1

    def test_position_alert_category(self, alert_manager, sample_state):
        """Test that position alerts have POSITION category."""
        sample_state.position_data.total_unrealized_pnl = Decimal("-600")

        alerts = alert_manager.check_alerts(sample_state)

        position_alerts = [a for a in alerts if a.category == AlertCategory.POSITION]
        assert len(position_alerts) == 1
