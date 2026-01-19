"""Tests for Alert dataclass creation and serialization."""

from __future__ import annotations

from datetime import UTC, datetime

from gpt_trader.monitoring.alert_types import Alert, AlertSeverity
from gpt_trader.utilities.datetime_helpers import utc_now


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
