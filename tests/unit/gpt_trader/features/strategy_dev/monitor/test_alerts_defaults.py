"""Tests for default alert factory."""

from __future__ import annotations

from gpt_trader.features.strategy_dev.monitor.alerts import (
    AlertSeverity,
    create_default_alerts,
)


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
