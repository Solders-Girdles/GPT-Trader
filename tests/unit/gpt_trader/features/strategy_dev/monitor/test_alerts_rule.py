"""Tests for AlertRule."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.features.strategy_dev.monitor.alerts import (
    AlertCondition,
    AlertRule,
    AlertSeverity,
)
from tests.unit.gpt_trader.features.strategy_dev.monitor.alerts_test_helpers import (
    create_test_snapshot,
)


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
