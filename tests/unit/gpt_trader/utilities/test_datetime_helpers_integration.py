"""Tests for datetime helper utilities (integration workflows)."""

from datetime import UTC, datetime, timedelta, timezone

from gpt_trader.utilities.datetime_helpers import (
    normalize_to_utc,
    to_iso_utc,
    utc_now,
    utc_now_iso,
    utc_timestamp,
)


class TestIntegrationWorkflow:
    """Test integration between different functions."""

    def test_roundtrip_datetime_to_iso_and_back(self) -> None:
        """Test roundtrip conversion: datetime -> iso -> datetime."""
        original = datetime(2024, 1, 15, 10, 30, 45, 123456, tzinfo=UTC)

        # Convert to ISO
        iso_str = to_iso_utc(original)

        # Parse back from ISO
        parsed = datetime.fromisoformat(iso_str.replace("+00:00", "+00:00"))

        # Normalize to UTC
        normalized = normalize_to_utc(parsed)

        assert normalized == original

    def test_all_functions_use_utc(self) -> None:
        """Test all functions consistently use UTC."""
        # Get current time using different methods
        dt_now = utc_now()
        iso_now = utc_now_iso()
        ts_now = utc_timestamp()

        # Parse ISO back to datetime
        dt_from_iso = datetime.fromisoformat(iso_now.replace("+00:00", "+00:00"))
        dt_from_ts = datetime.fromtimestamp(ts_now, tz=UTC)

        # Normalize all to UTC
        dt_from_iso_norm = normalize_to_utc(dt_from_iso)

        # All should be timezone-aware and in UTC
        assert dt_now.tzinfo == UTC
        assert dt_from_iso_norm.tzinfo == UTC
        assert dt_from_ts.tzinfo == UTC

        # All should be close in time
        assert abs((dt_from_iso_norm - dt_now).total_seconds()) < 1.0
        assert abs((dt_from_ts - dt_now).total_seconds()) < 1.0

    def test_timezone_conversion_consistency(self) -> None:
        """Test timezone conversion is consistent across functions."""
        # Create datetime in different timezone
        est = timezone(timedelta(hours=-5))
        original = datetime(2024, 1, 15, 10, 30, 45, tzinfo=est)

        # Convert using different methods
        iso_result = to_iso_utc(original)
        normalized_result = normalize_to_utc(original)
        iso_from_normalized = to_iso_utc(normalized_result)

        # All should represent the same moment in time
        assert iso_result == iso_from_normalized
        assert normalized_result.tzinfo == UTC

        # Parse ISO and compare
        parsed_iso = datetime.fromisoformat(iso_result.replace("+00:00", "+00:00"))
        parsed_iso_norm = normalize_to_utc(parsed_iso)

        assert parsed_iso_norm == normalized_result
