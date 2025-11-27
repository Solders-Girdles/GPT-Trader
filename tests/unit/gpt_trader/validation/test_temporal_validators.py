"""Tests for temporal validators."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.temporal_validators import DateValidator


class TestDateValidator:
    """Tests for DateValidator."""

    def test_validates_date_object(self) -> None:
        validator = DateValidator()
        test_date = date(2024, 1, 15)
        assert validator.validate(test_date, "start_date") == test_date

    def test_validates_datetime_object(self) -> None:
        validator = DateValidator()
        test_datetime = datetime(2024, 1, 15, 10, 30, 0)
        assert validator.validate(test_datetime, "timestamp") == test_datetime

    def test_validates_iso_date_string(self) -> None:
        validator = DateValidator()
        result = validator.validate("2024-01-15", "date")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_validates_iso_datetime_string(self) -> None:
        validator = DateValidator()
        result = validator.validate("2024-01-15T10:30:00", "timestamp")
        assert isinstance(result, datetime)
        assert result.hour == 10
        assert result.minute == 30

    def test_validates_within_min_date(self) -> None:
        min_date = date(2024, 1, 1)
        validator = DateValidator(min_date=min_date)
        test_date = date(2024, 6, 15)
        assert validator.validate(test_date, "date") == test_date

    def test_validates_at_min_date_boundary(self) -> None:
        min_date = date(2024, 1, 1)
        validator = DateValidator(min_date=min_date)
        # Exact boundary should pass (not strict <)
        assert validator.validate(date(2024, 1, 2), "date") == date(2024, 1, 2)

    def test_rejects_before_min_date(self) -> None:
        min_date = date(2024, 1, 1)
        validator = DateValidator(min_date=min_date)
        with pytest.raises(ValidationError, match="must be after"):
            validator.validate(date(2023, 12, 31), "start_date")

    def test_validates_within_max_date(self) -> None:
        max_date = date(2024, 12, 31)
        validator = DateValidator(max_date=max_date)
        test_date = date(2024, 6, 15)
        assert validator.validate(test_date, "date") == test_date

    def test_rejects_after_max_date(self) -> None:
        max_date = date(2024, 12, 31)
        validator = DateValidator(max_date=max_date)
        with pytest.raises(ValidationError, match="must be before"):
            validator.validate(date(2025, 1, 1), "end_date")

    def test_validates_within_range(self) -> None:
        min_date = date(2024, 1, 1)
        max_date = date(2024, 12, 31)
        validator = DateValidator(min_date=min_date, max_date=max_date)
        test_date = date(2024, 6, 15)
        assert validator.validate(test_date, "date") == test_date

    def test_rejects_invalid_iso_string(self) -> None:
        validator = DateValidator()
        with pytest.raises(ValidationError, match="must be a valid date/datetime"):
            validator.validate("not-a-date", "date")

    def test_rejects_partial_date_string(self) -> None:
        validator = DateValidator()
        with pytest.raises(ValidationError, match="must be a valid date/datetime"):
            validator.validate("2024-13-45", "date")

    def test_rejects_non_date_type(self) -> None:
        validator = DateValidator()
        with pytest.raises(ValidationError, match="must be a date or datetime"):
            validator.validate(12345, "date")

    def test_rejects_none(self) -> None:
        validator = DateValidator()
        with pytest.raises(ValidationError, match="must be a date or datetime"):
            validator.validate(None, "date")

    def test_field_name_in_error(self) -> None:
        validator = DateValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("bad", "trade_date")
        assert exc_info.value.context["field"] == "trade_date"

    def test_datetime_comparison_with_datetime_bounds(self) -> None:
        # Use datetime bounds when validating datetime values
        min_dt = datetime(2024, 1, 1, 0, 0, 0)
        max_dt = datetime(2024, 12, 31, 23, 59, 59)
        validator = DateValidator(min_date=min_dt, max_date=max_dt)
        test_dt = datetime(2024, 6, 15, 12, 0, 0)
        assert validator.validate(test_dt, "timestamp") == test_dt
