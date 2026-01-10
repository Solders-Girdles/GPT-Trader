"""Tests for AlertManager service."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.tui.services.alert_manager import (
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
def sample_state():
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

    def test_alert_triggered_on_condition(self, alert_manager, mock_app, sample_state):
        """Test that alerts are triggered when conditions are met."""
        # Set up a condition that will trigger
        sample_state.system_data.connection_status = "DISCONNECTED"

        alerts = alert_manager.check_alerts(sample_state)

        assert len(alerts) == 1
        assert alerts[0].rule_id == "connection_lost"
        assert alerts[0].severity == AlertSeverity.ERROR
        mock_app.notify.assert_called_once()

    def test_alert_cooldown(self, alert_manager, mock_app, sample_state):
        """Test that cooldown prevents repeated alerts."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        # First alert should trigger
        alerts1 = alert_manager.check_alerts(sample_state)
        assert len(alerts1) == 1

        # Second check should be blocked by cooldown
        alerts2 = alert_manager.check_alerts(sample_state)
        assert len(alerts2) == 0

        # Notify should only be called once
        assert mock_app.notify.call_count == 1

    def test_no_alert_when_condition_not_met(self, alert_manager, mock_app, sample_state):
        """Test that no alerts trigger when conditions aren't met."""
        # sample_state has all good values by default
        alerts = alert_manager.check_alerts(sample_state)

        # No alerts should trigger (state is healthy)
        assert len(alerts) == 0
        mock_app.notify.assert_not_called()

    def test_rate_limit_alert(self, alert_manager, mock_app, sample_state):
        """Test rate limit warning alert."""
        sample_state.system_data.rate_limit_usage = "85%"

        alerts = alert_manager.check_alerts(sample_state)

        # Should trigger rate limit warning
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

        # First alert
        alert_manager.check_alerts(sample_state)
        assert mock_app.notify.call_count == 1

        # Reset cooldowns
        alert_manager.reset_cooldowns()

        # Should trigger again after reset
        alert_manager.check_alerts(sample_state)
        assert mock_app.notify.call_count == 2

    def test_disabled_rule_not_checked(self, alert_manager, mock_app, sample_state):
        """Test that disabled rules are not checked."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alert_manager.disable_rule("connection_lost")
        alerts = alert_manager.check_alerts(sample_state)

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

    def test_alert_has_category(self, alert_manager, sample_state):
        """Test that triggered alerts have a category."""
        sample_state.system_data.connection_status = "DISCONNECTED"

        alerts = alert_manager.check_alerts(sample_state)

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

    def test_filter_history_by_category(self, alert_manager, sample_state):
        """Test filtering alert history by category."""
        # Trigger a SYSTEM category alert
        sample_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(sample_state)

        # Get all history
        all_history = alert_manager.get_history()
        assert len(all_history) == 1

        # Filter by SYSTEM category
        system_history = alert_manager.get_history(categories={AlertCategory.SYSTEM})
        assert len(system_history) == 1

        # Filter by TRADE category (should be empty)
        trade_history = alert_manager.get_history(categories={AlertCategory.TRADE})
        assert len(trade_history) == 0

    def test_filter_history_by_severity(self, alert_manager, sample_state):
        """Test filtering alert history by minimum severity."""
        # Trigger an ERROR severity alert
        sample_state.system_data.connection_status = "DISCONNECTED"
        alert_manager.check_alerts(sample_state)

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
        # Initially all counts should be 0
        counts = alert_manager.get_category_counts()
        assert all(count == 0 for count in counts.values())

        # Trigger a SYSTEM alert
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
        # Set up resilience data with circuit breaker open
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

        # Critical should fire, but not degraded
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
        # Don't set execution_data or resilience_data
        # Should not raise exceptions
        alerts = alert_manager.check_alerts(sample_state)

        # Execution alerts should not fire without data
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


class TestTradeAlerts:
    """Test suite for trade-related alert rules (stale, failed, expired orders)."""

    def test_stale_orders_rules_registered(self, alert_manager):
        """Test that trade alert rules are registered."""
        status = alert_manager.get_rule_status()
        assert "stale_open_orders" in status
        assert "failed_orders" in status
        assert "expired_orders" in status

    def test_stale_orders_alert_triggers(self, alert_manager, mock_app, sample_state):
        """Test stale orders alert triggers when order is old enough."""
        import time

        from gpt_trader.tui.thresholds import DEFAULT_ORDER_THRESHOLDS
        from gpt_trader.tui.types import ActiveOrders, Order

        # Create an order older than the warning threshold
        stale_age = DEFAULT_ORDER_THRESHOLDS.age_warn + 10
        old_order = Order(
            order_id="stale-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="OPEN",
            creation_time=time.time() - stale_age,
        )
        sample_state.order_data = ActiveOrders(orders=[old_order])

        alerts = alert_manager.check_alerts(sample_state)

        stale_alerts = [a for a in alerts if a.rule_id == "stale_open_orders"]
        assert len(stale_alerts) == 1
        assert stale_alerts[0].severity == AlertSeverity.WARNING
        assert stale_alerts[0].category == AlertCategory.TRADE
        assert "BTC-USD" in stale_alerts[0].message

    def test_stale_orders_no_alert_when_fresh(self, alert_manager, mock_app, sample_state):
        """Test no stale orders alert when orders are fresh."""
        import time

        from gpt_trader.tui.types import ActiveOrders, Order

        # Create a fresh order (just created)
        fresh_order = Order(
            order_id="fresh-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="OPEN",
            creation_time=time.time() - 5,  # 5 seconds old
        )
        sample_state.order_data = ActiveOrders(orders=[fresh_order])

        alerts = alert_manager.check_alerts(sample_state)

        stale_alerts = [a for a in alerts if a.rule_id == "stale_open_orders"]
        assert len(stale_alerts) == 0

    def test_stale_orders_ignores_filled_orders(self, alert_manager, mock_app, sample_state):
        """Test stale orders alert ignores non-open orders."""
        import time

        from gpt_trader.tui.thresholds import DEFAULT_ORDER_THRESHOLDS
        from gpt_trader.tui.types import ActiveOrders, Order

        # Old but already filled - should not trigger
        stale_age = DEFAULT_ORDER_THRESHOLDS.age_warn + 100
        filled_order = Order(
            order_id="filled-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="FILLED",
            creation_time=time.time() - stale_age,
        )
        sample_state.order_data = ActiveOrders(orders=[filled_order])

        alerts = alert_manager.check_alerts(sample_state)

        stale_alerts = [a for a in alerts if a.rule_id == "stale_open_orders"]
        assert len(stale_alerts) == 0

    def test_failed_orders_alert_triggers(self, alert_manager, mock_app, sample_state):
        """Test failed orders alert triggers on REJECTED status."""
        from gpt_trader.tui.types import ActiveOrders, Order

        rejected_order = Order(
            order_id="rejected-1",
            symbol="ETH-USD",
            side="SELL",
            quantity=Decimal("1.0"),
            price=Decimal("3000"),
            status="REJECTED",
        )
        sample_state.order_data = ActiveOrders(orders=[rejected_order])

        alerts = alert_manager.check_alerts(sample_state)

        failed_alerts = [a for a in alerts if a.rule_id == "failed_orders"]
        assert len(failed_alerts) == 1
        assert failed_alerts[0].severity == AlertSeverity.ERROR
        assert failed_alerts[0].category == AlertCategory.TRADE
        assert "ETH-USD" in failed_alerts[0].message
        assert "rejected" in failed_alerts[0].message.lower()

    def test_failed_orders_alert_triggers_on_failed_status(
        self, alert_manager, mock_app, sample_state
    ):
        """Test failed orders alert triggers on FAILED status."""
        from gpt_trader.tui.types import ActiveOrders, Order

        failed_order = Order(
            order_id="failed-1",
            symbol="SOL-USD",
            side="BUY",
            quantity=Decimal("10.0"),
            price=Decimal("100"),
            status="FAILED",
        )
        sample_state.order_data = ActiveOrders(orders=[failed_order])

        alerts = alert_manager.check_alerts(sample_state)

        failed_alerts = [a for a in alerts if a.rule_id == "failed_orders"]
        assert len(failed_alerts) == 1
        assert "SOL-USD" in failed_alerts[0].message

    def test_failed_orders_no_alert_on_cancelled(self, alert_manager, mock_app, sample_state):
        """Test failed orders alert does NOT trigger on CANCELLED (user-initiated)."""
        from gpt_trader.tui.types import ActiveOrders, Order

        cancelled_order = Order(
            order_id="cancelled-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            status="CANCELLED",
        )
        sample_state.order_data = ActiveOrders(orders=[cancelled_order])

        alerts = alert_manager.check_alerts(sample_state)

        failed_alerts = [a for a in alerts if a.rule_id == "failed_orders"]
        assert len(failed_alerts) == 0

    def test_expired_orders_alert_triggers(self, alert_manager, mock_app, sample_state):
        """Test expired orders alert triggers on EXPIRED status."""
        from gpt_trader.tui.types import ActiveOrders, Order

        expired_order = Order(
            order_id="expired-1",
            symbol="BTC-USD",
            side="SELL",
            quantity=Decimal("0.2"),
            price=Decimal("50000"),
            status="EXPIRED",
        )
        sample_state.order_data = ActiveOrders(orders=[expired_order])

        alerts = alert_manager.check_alerts(sample_state)

        expired_alerts = [a for a in alerts if a.rule_id == "expired_orders"]
        assert len(expired_alerts) == 1
        assert expired_alerts[0].severity == AlertSeverity.WARNING
        assert expired_alerts[0].category == AlertCategory.TRADE
        assert "BTC-USD" in expired_alerts[0].message
        assert "expired" in expired_alerts[0].message.lower()

    def test_expired_orders_no_alert_when_none_expired(self, alert_manager, mock_app, sample_state):
        """Test no expired orders alert when no orders are expired."""
        from gpt_trader.tui.types import ActiveOrders, Order

        open_order = Order(
            order_id="open-1",
            symbol="BTC-USD",
            side="BUY",
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            status="OPEN",
        )
        sample_state.order_data = ActiveOrders(orders=[open_order])

        alerts = alert_manager.check_alerts(sample_state)

        expired_alerts = [a for a in alerts if a.rule_id == "expired_orders"]
        assert len(expired_alerts) == 0

    def test_multiple_expired_orders_message(self, alert_manager, mock_app, sample_state):
        """Test expired orders alert message when multiple orders expired."""
        from gpt_trader.tui.types import ActiveOrders, Order

        expired_orders = [
            Order(
                order_id="exp-1",
                symbol="BTC-USD",
                side="BUY",
                quantity=Decimal("0.1"),
                price=Decimal("50000"),
                status="EXPIRED",
            ),
            Order(
                order_id="exp-2",
                symbol="ETH-USD",
                side="SELL",
                quantity=Decimal("1.0"),
                price=Decimal("3000"),
                status="EXPIRED",
            ),
        ]
        sample_state.order_data = ActiveOrders(orders=expired_orders)

        alerts = alert_manager.check_alerts(sample_state)

        expired_alerts = [a for a in alerts if a.rule_id == "expired_orders"]
        assert len(expired_alerts) == 1
        assert "2 orders expired" in expired_alerts[0].message


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
        sample_state.system_data.validation_escalated = True  # Already escalated
        sample_state.system_data.validation_failures = {"mark_staleness": 5}

        alerts = alert_manager.check_alerts(sample_state)

        # Escalation alert should fire, but not failures warning
        escalation_alerts = [a for a in alerts if a.rule_id == "validation_escalation"]
        failure_alerts = [a for a in alerts if a.rule_id == "validation_failures"]
        assert len(escalation_alerts) == 1
        assert len(failure_alerts) == 0

    def test_validation_alerts_handle_missing_data(self, alert_manager, mock_app, sample_state):
        """Test validation alerts gracefully handle missing data."""
        # Don't set validation_failures or validation_escalated
        # SystemStatus defaults should handle this
        alerts = alert_manager.check_alerts(sample_state)

        validation_alerts = [
            a for a in alerts if a.rule_id in {"validation_escalation", "validation_failures"}
        ]
        assert len(validation_alerts) == 0
