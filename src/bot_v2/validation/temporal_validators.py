"""
Temporal validators for dates and datetimes.
"""

from datetime import date, datetime
from typing import Any

from bot_v2.errors import ValidationError

from .base_validators import Validator


class DateValidator(Validator):
    """Validate date/datetime"""

    def __init__(
        self,
        min_date: date | datetime | None = None,
        max_date: date | datetime | None = None,
        error_message: str | None = None,
    ) -> None:
        super().__init__(error_message)
        self.min_date = min_date
        self.max_date = max_date

    def validate(self, value: Any, field_name: str = "date") -> date | datetime:
        # Convert string to datetime if needed
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                raise ValidationError(
                    f"{field_name} must be a valid date/datetime", field=field_name, value=value
                )

        if not isinstance(value, (date, datetime)):
            raise ValidationError(
                f"{field_name} must be a date or datetime", field=field_name, value=value
            )

        if self.min_date and value < self.min_date:
            raise ValidationError(
                f"{field_name} must be after {self.min_date}", field=field_name, value=value
            )

        if self.max_date and value > self.max_date:
            raise ValidationError(
                f"{field_name} must be before {self.max_date}", field=field_name, value=value
            )

        return value
