"""Tests for alerts module."""

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.features.strategy_dev.monitor.alerts import (
    AlertCondition,
    AlertManager,
    AlertRule,
    AlertSeverity,
    create_default_alerts,
)
from gpt_trader.features.strategy_dev.monitor.metrics import PerformanceSnapshot


def create_test_snapshot(**kwargs) -> PerformanceSnapshot:
    """Create a test snapshot with defaults."""
    defaults = {
        "timestamp": datetime.now(),
        "equity": Decimal("10000"),
        "cash": Decimal("5000"),
        "positions_value": Decimal("5000"),
        "total_return": 0.0,
        "daily_return": 0.0,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "drawdown": 0.0,
        "max_drawdown": 0.0,
        "volatility": 0.0,
        "open_positions": 0,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "current_regime": "UNKNOWN",
        "regime_confidence": 0.0,
    }
    defaults.update(kwargs)
    return PerformanceSnapshot(**defaults)


class TestAlertRule:
    """Tests for AlertRule."""

    def test_drawdown_rule(self):
        """Test drawdown threshold rule."""
        rule = AlertRule(
            name="test_drawdown",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
            severity=AlertSeverity.WARNING,
        )

        # Should not trigger below threshold
        snapshot1 = create_test_snapshot(drawdown=0.05)
        result1 = rule.check(snapshot1)
        assert result1 is None

        # Should trigger at threshold
        snapshot2 = create_test_snapshot(drawdown=0.15)
        result2 = rule.check(snapshot2)
        assert result2 is not None
        assert result2.severity == AlertSeverity.WARNING

    def test_volatility_rule(self):
        """Test volatility spike rule."""
        rule = AlertRule(
            name="vol_spike",
            condition=AlertCondition.VOLATILITY_SPIKE,
            threshold=0.30,
        )

        snapshot = create_test_snapshot(volatility=0.40)
        result = rule.check(snapshot)

        assert result is not None
        assert "Volatility" in result.message

    def test_regime_change_rule(self):
        """Test regime change detection."""
        rule = AlertRule(
            name="regime_change",
            condition=AlertCondition.REGIME_CHANGE,
            severity=AlertSeverity.INFO,
        )

        prev = create_test_snapshot(current_regime="BULL_QUIET")
        curr = create_test_snapshot(current_regime="CRISIS")

        result = rule.check(curr, prev)

        assert result is not None
        assert "BULL_QUIET" in result.message
        assert "CRISIS" in result.message

    def test_cooldown(self):
        """Test alert cooldown."""
        rule = AlertRule(
            name="test_cooldown",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
            cooldown_seconds=300,
        )

        snapshot = create_test_snapshot(drawdown=0.15)

        # First trigger
        result1 = rule.check(snapshot)
        assert result1 is not None

        # Should not trigger again (cooldown)
        result2 = rule.check(snapshot)
        assert result2 is None

    def test_custom_rule(self):
        """Test custom condition rule."""

        def custom_check(snapshot):
            return float(snapshot.equity) < 9000

        def custom_message(snapshot):
            return f"Equity critically low: {snapshot.equity}"

        rule = AlertRule(
            name="custom_equity",
            condition=AlertCondition.CUSTOM,
            custom_check=custom_check,
            custom_message=custom_message,
            severity=AlertSeverity.CRITICAL,
        )

        snapshot = create_test_snapshot(equity=Decimal("8500"))
        result = rule.check(snapshot)

        assert result is not None
        assert result.severity == AlertSeverity.CRITICAL
        assert "8500" in result.message

    def test_disabled_rule(self):
        """Test disabled rule doesn't trigger."""
        rule = AlertRule(
            name="disabled",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.01,
            enabled=False,
        )

        snapshot = create_test_snapshot(drawdown=0.50)
        result = rule.check(snapshot)

        assert result is None


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def manager(self):
        """Create alert manager."""
        return AlertManager()

    def test_add_rule(self, manager):
        """Test adding rules."""
        rule = AlertRule(
            name="test",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
        )

        manager.add_rule(rule)

        assert "test" in manager.rules

    def test_remove_rule(self, manager):
        """Test removing rules."""
        rule = AlertRule(
            name="test",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
        )

        manager.add_rule(rule)
        removed = manager.remove_rule("test")

        assert removed is True
        assert "test" not in manager.rules

    def test_process_snapshot(self, manager):
        """Test processing snapshots."""
        rule = AlertRule(
            name="drawdown",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
            cooldown_seconds=0,  # No cooldown for test
        )
        manager.add_rule(rule)

        snapshot = create_test_snapshot(drawdown=0.15)
        alerts = manager.process(snapshot)

        assert len(alerts) == 1
        assert alerts[0].rule_name == "drawdown"

    def test_callback(self, manager):
        """Test alert callbacks."""
        alerts_received = []

        def callback(alert):
            alerts_received.append(alert)

        manager.on_alert(callback)

        rule = AlertRule(
            name="test",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
            cooldown_seconds=0,
        )
        manager.add_rule(rule)

        snapshot = create_test_snapshot(drawdown=0.15)
        manager.process(snapshot)

        assert len(alerts_received) == 1

    def test_history(self, manager):
        """Test alert history."""
        rule = AlertRule(
            name="test",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
            cooldown_seconds=0,
        )
        manager.add_rule(rule)

        for _ in range(3):
            snapshot = create_test_snapshot(drawdown=0.15)
            manager.process(snapshot)
            rule.reset()  # Reset cooldown

        recent = manager.get_recent_alerts(limit=10)
        assert len(recent) == 3

    def test_get_alert_summary(self, manager):
        """Test alert summary."""
        rule = AlertRule(
            name="test",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
            cooldown_seconds=0,
        )
        manager.add_rule(rule)

        snapshot = create_test_snapshot(drawdown=0.15)
        manager.process(snapshot)

        summary = manager.get_alert_summary()

        assert summary["total_rules"] == 1
        assert summary["alerts_24h"] >= 1

    def test_clear_history(self, manager):
        """Test clearing history."""
        rule = AlertRule(
            name="test",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            threshold=0.10,
            cooldown_seconds=0,
        )
        manager.add_rule(rule)

        snapshot = create_test_snapshot(drawdown=0.15)
        manager.process(snapshot)

        cleared = manager.clear_history()

        assert cleared == 1
        assert len(manager.history) == 0


class TestDefaultAlerts:
    """Tests for default alert factory."""

    def test_create_default_alerts(self):
        """Test creating default alerts."""
        alerts = create_default_alerts()

        assert len(alerts) >= 5

        names = [a.name for a in alerts]
        assert "critical_drawdown" in names
        assert "regime_change" in names

    def test_default_severities(self):
        """Test default alert severities."""
        alerts = create_default_alerts()

        for alert in alerts:
            if "critical" in alert.name:
                assert alert.severity == AlertSeverity.CRITICAL
