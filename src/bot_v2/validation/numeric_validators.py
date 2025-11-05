"""
Numeric validators for validating numbers, percentages, and ranges.
"""

from typing import Any

from bot_v2.errors import ValidationError

from .base_validators import Validator


class PositiveNumberValidator(Validator):
    """Validate positive number"""

    def __init__(self, allow_zero: bool = False, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.allow_zero = allow_zero

    def validate(self, value: Any, field_name: str = "value") -> float:
        numeric_value: float
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be a number", field=field_name, value=value)

        if self.allow_zero and numeric_value < 0:
            raise ValidationError(f"{field_name} must be >= 0", field=field_name, value=value)
        elif not self.allow_zero and numeric_value <= 0:
            raise ValidationError(f"{field_name} must be > 0", field=field_name, value=value)

        return numeric_value


class PercentageValidator(Validator):
    """Validate percentage (0-100 or 0-1)"""

    def __init__(self, as_decimal: bool = True, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.as_decimal = as_decimal

    def validate(self, value: Any, field_name: str = "percentage") -> float:
        numeric_value: float
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be a number", field=field_name, value=value)

        if self.as_decimal:
            if not 0 <= numeric_value <= 1:
                raise ValidationError(
                    f"{field_name} must be between 0 and 1",
                    field=field_name,
                    value=value,
                )
        else:
            if not 0 <= numeric_value <= 100:
                raise ValidationError(
                    f"{field_name} must be between 0 and 100",
                    field=field_name,
                    value=value,
                )
            numeric_value = numeric_value / 100  # Convert to decimal

        return numeric_value
