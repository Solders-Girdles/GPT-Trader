"""Tests for alert types module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.utilities.datetime_helpers import utc_now


class TestAlertSeverityNumericLevel:
    """Tests for AlertSeverity.numeric_level property."""

    def test_debug_level(self) -> None:
        assert AlertSeverity.DEBUG.numeric_level == 10

    def test_info_level(self) -> None:
        assert AlertSeverity.INFO.numeric_level == 20

    def test_warning_level(self) -> None:
        assert AlertSeverity.WARNING.numeric_level == 30

    def test_error_level(self) -> None:
        assert AlertSeverity.ERROR.numeric_level == 40

    def test_critical_level(self) -> None:
        assert AlertSeverity.CRITICAL.numeric_level == 50

    def test_levels_are_ordered(self) -> None:
        assert AlertSeverity.DEBUG.numeric_level < AlertSeverity.INFO.numeric_level
        assert AlertSeverity.INFO.numeric_level < AlertSeverity.WARNING.numeric_level
        assert AlertSeverity.WARNING.numeric_level < AlertSeverity.ERROR.numeric_level
        assert AlertSeverity.ERROR.numeric_level < AlertSeverity.CRITICAL.numeric_level


class TestAlertSeverityCoerce:
    """Tests for AlertSeverity.coerce class method."""

    def test_coerce_from_enum(self) -> None:
        result = AlertSeverity.coerce(AlertSeverity.ERROR)
        assert result is AlertSeverity.ERROR

    def test_coerce_from_numeric_10(self) -> None:
        result = AlertSeverity.coerce(10)
        assert result is AlertSeverity.DEBUG

    def test_coerce_from_numeric_20(self) -> None:
        result = AlertSeverity.coerce(20)
        assert result is AlertSeverity.INFO

    def test_coerce_from_numeric_30(self) -> None:
        result = AlertSeverity.coerce(30)
        assert result is AlertSeverity.WARNING

    def test_coerce_from_numeric_40(self) -> None:
        result = AlertSeverity.coerce(40)
        assert result is AlertSeverity.ERROR

    def test_coerce_from_numeric_50(self) -> None:
        result = AlertSeverity.coerce(50)
        assert result is AlertSeverity.CRITICAL

    def test_coerce_from_numeric_high_defaults_critical(self) -> None:
        result = AlertSeverity.coerce(100)
        assert result is AlertSeverity.CRITICAL

    def test_coerce_from_numeric_low_defaults_debug(self) -> None:
        result = AlertSeverity.coerce(5)
        assert result is AlertSeverity.DEBUG

    def test_coerce_from_string_value(self) -> None:
        result = AlertSeverity.coerce("warning")
        assert result is AlertSeverity.WARNING

    def test_coerce_from_string_name(self) -> None:
        result = AlertSeverity.coerce("WARNING")
        assert result is AlertSeverity.WARNING

    def test_coerce_strips_whitespace(self) -> None:
        result = AlertSeverity.coerce("  error  ")
        assert result is AlertSeverity.ERROR

    def test_coerce_case_insensitive(self) -> None:
        result = AlertSeverity.coerce("CrItIcAl")
        assert result is AlertSeverity.CRITICAL

    def test_coerce_raises_for_unknown_string(self) -> None:
        with pytest.raises(ValueError, match="Unknown alert severity"):
            AlertSeverity.coerce("invalid_severity")


class TestAlertCreation:
    """Tests for Alert dataclass creation."""

    def test_minimal_creation(self) -> None:
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test",
        )
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"

    def test_auto_generates_id(self) -> None:
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )
        assert alert.alert_id is not None
        assert len(alert.alert_id) == 32  # UUID hex

    def test_different_alerts_have_different_ids(self) -> None:
        alert1 = Alert(severity=AlertSeverity.INFO, title="A", message="A")
        alert2 = Alert(severity=AlertSeverity.INFO, title="B", message="B")
        assert alert1.alert_id != alert2.alert_id

    def test_auto_sets_created_at(self) -> None:
        before = utc_now()
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        after = utc_now()
        assert before <= alert.created_at <= after

    def test_last_seen_at_defaults_to_created_at(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.last_seen_at == alert.created_at

    def test_occurrences_defaults_to_1(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.occurrences == 1

    def test_metadata_defaults_empty(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.metadata == {}

    def test_context_defaults_empty(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.context == {}

    def test_details_defaults_empty(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.details == {}

    def test_resolved_at_defaults_none(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.resolved_at is None

    def test_acknowledged_defaults_false(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        assert alert.acknowledged is False

    def test_created_at_override(self) -> None:
        custom_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            created_at_override=custom_time,
        )
        assert alert.created_at == custom_time


class TestAlertProperties:
    """Tests for Alert properties."""

    def test_id_property_returns_alert_id(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        with pytest.warns(DeprecationWarning, match="Alert.id is deprecated"):
            assert alert.id == alert.alert_id

    def test_timestamp_property_returns_created_at(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        with pytest.warns(DeprecationWarning, match="Alert.timestamp is deprecated"):
            assert alert.timestamp == alert.created_at


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


class TestAlertToDict:
    """Tests for Alert.to_dict method."""

    def test_returns_dict(self) -> None:
        alert = Alert(severity=AlertSeverity.INFO, title="Test", message="Test")
        result = alert.to_dict()
        assert isinstance(result, dict)

    def test_includes_all_fields(self) -> None:
        alert = Alert(
            severity=AlertSeverity.WARNING,
            title="Test Title",
            message="Test Message",
            source="test_source",
            category="test_category",
            component="test_component",
        )
        result = alert.to_dict()

        assert result["id"] == alert.alert_id
        assert result["severity"] == "warning"
        assert result["title"] == "Test Title"
        assert result["message"] == "Test Message"
        assert result["source"] == "test_source"
        assert result["category"] == "test_category"
        assert result["component"] == "test_component"

    def test_serializes_datetimes_to_iso_format(self) -> None:
        custom_time = datetime(2024, 6, 15, 10, 30, 0, tzinfo=UTC)
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            created_at_override=custom_time,
        )
        result = alert.to_dict()
        assert result["created_at"] == "2024-06-15T10:30:00+00:00"


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
        # Create alert 60 seconds ago
        past_time = utc_now() - timedelta(seconds=60)
        alert = Alert(
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            created_at_override=past_time,
        )
        age = alert.age_minutes()
        assert 0.9 < age < 1.1  # Should be about 1 minute

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
        assert age == 30.0  # Exactly 30 minutes
