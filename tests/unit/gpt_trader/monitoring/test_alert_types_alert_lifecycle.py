"""Tests for Alert lifecycle methods."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.utilities.datetime_helpers import utc_now


class TestAlertTouch:
    """Tests for Alert.touch method."""

    def test_touch_increments_occurrences(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.occurrences == 1
        alert.touch()
        assert alert.occurrences == 2
        alert.touch()
        assert alert.occurrences == 3

    def test_touch_updates_last_seen_at(self) -> None:
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            created_at_override=custom_time,
        )
        before = utc_now()
        alert.touch()
        after = utc_now()
        assert before <= alert.last_seen_at <= after
        assert alert.last_seen_at > custom_time

    def test_touch_with_context_updates_context(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        alert.touch(context={"key": "value"})
        assert alert.context == {"key": "value"}

    def test_touch_with_metadata_updates_metadata(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        alert.touch(metadata={"meta_key": "meta_value"})
        assert alert.metadata == {"meta_key": "meta_value"}

    def test_touch_merges_context(self) -> None:
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            context={"existing": "data"},
        )
        alert.touch(context={"new": "value"})
        assert alert.context == {"existing": "data", "new": "value"}


class TestAlertMarkResolved:
    """Tests for Alert.mark_resolved method."""

    def test_sets_resolved_at(self) -> None:
        alert = Alert(severity=AlertSeverity.ERROR, title="Test", message="Test")
        assert alert.resolved_at is None

        before = utc_now()
        alert.mark_resolved()
        after = utc_now()

        assert alert.resolved_at is not None
        assert before <= alert.resolved_at <= after


class TestAlertAcknowledge:
    """Tests for Alert.acknowledge method."""

    def test_sets_acknowledged_flag(self) -> None:
        alert = Alert(severity=AlertSeverity.ERROR, title="Test", message="Test")
        assert alert.acknowledged is False

        alert.acknowledge()

        assert alert.acknowledged is True


class TestAlertIsActive:
    """Tests for Alert.is_active method."""

    def test_active_when_not_resolved(self) -> None:
        alert = Alert(severity=AlertSeverity.ERROR, title="Test", message="Test")
        assert alert.is_active() is True

    def test_not_active_when_resolved(self) -> None:
        alert = Alert(severity=AlertSeverity.ERROR, title="Test", message="Test")
        alert.mark_resolved()
        assert alert.is_active() is False


class TestAlertAgeMinutes:
    """Tests for Alert.age_minutes method."""

    def test_age_for_unresolved_alert(self) -> None:
        past_time = utc_now() - timedelta(seconds=60)
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            created_at_override=past_time,
        )
        age = alert.age_minutes()
        assert 0.9 < age < 1.1

    def test_age_for_resolved_alert(self) -> None:
        past_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        resolved_time = datetime(2024, 1, 1, 12, 30, 0, tzinfo=UTC)

        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            created_at_override=past_time,
        )
        alert.resolved_at = resolved_time

        age = alert.age_minutes()
        assert age == 30.0
