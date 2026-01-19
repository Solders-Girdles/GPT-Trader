"""Tests for AlertManager."""

from __future__ import annotations

import pytest

from gpt_trader.features.strategy_dev.monitor.alerts import (
    AlertCondition,
    AlertManager,
    AlertRule,
)
from tests.unit.gpt_trader.features.strategy_dev.monitor.alerts_test_helpers import (
    create_test_snapshot,
)


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
