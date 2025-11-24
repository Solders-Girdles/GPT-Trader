from __future__ import annotations

from datetime import timedelta

from gpt_trader.monitoring.alert_types import AlertSeverity


def test_alert_manager_deduplicates_within_window(alert_manager, frozen_time):
    alert = alert_manager.create_alert(
        severity=AlertSeverity.WARNING,
        component="System",
        message="High CPU",
    )
    assert alert is not None

    duplicate = alert_manager.create_alert(
        severity=AlertSeverity.WARNING,
        component="System",
        message="High CPU",
    )
    assert duplicate is None

    frozen_time.tick(delta=timedelta(seconds=alert_manager.dedup_window_seconds + 1))
    replay = alert_manager.create_alert(
        severity=AlertSeverity.WARNING,
        component="System",
        message="High CPU",
    )
    assert replay is not None


def test_alert_manager_acknowledge_and_resolve(alert_manager):
    alert = alert_manager.create_alert(
        severity=AlertSeverity.ERROR,
        component="RiskEngine",
        message="Risk alert",
    )
    assert alert is not None

    assert alert_manager.acknowledge_alert(alert.alert_id)
    active = alert_manager.get_active_alerts()
    assert active[0].acknowledged is True

    assert alert_manager.resolve_alert(alert.alert_id)
    assert alert.alert_id not in {a.alert_id for a in alert_manager.get_active_alerts()}


def test_alert_manager_summary(alert_manager):
    critical = alert_manager.create_alert(
        severity=AlertSeverity.CRITICAL,
        component="Gateway",
        message="Gateway down",
    )
    error = alert_manager.create_alert(
        severity=AlertSeverity.ERROR,
        component="System",
        message="High error rate",
    )
    warning = alert_manager.create_alert(
        severity=AlertSeverity.WARNING,
        component="System",
        message="High CPU",
    )

    assert None not in (critical, error, warning)

    summary = alert_manager.get_alert_summary()
    assert summary["total_active"] == 3
    assert summary["critical"] == 1
    assert summary["error"] == 1
    assert summary["warning"] == 1
