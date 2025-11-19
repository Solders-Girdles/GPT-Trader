"""
Type-based validators for common validation patterns.
"""

import re
from typing import Any

from bot_v2.errors import ValidationError

from .base_validators import Validator


class TypeValidator(Validator):
    """Validate that value is of specific type"""

    def __init__(self, expected_type: type, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.expected_type = expected_type

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if not isinstance(value, self.expected_type):
            msg = (
                self.error_message or f"{field_name} must be of type {self.expected_type.__name__}"
            )
            raise ValidationError(msg, field=field_name, value=value)
        return value


class RangeValidator(Validator):
    """Validate that value is within range"""

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        inclusive: bool = True,
        error_message: str | None = None,
    ) -> None:
        super().__init__(error_message)
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if self.min_value is not None:
            if self.inclusive and value < self.min_value:
                msg = self.error_message or f"{field_name} must be >= {self.min_value}"
                raise ValidationError(msg, field=field_name, value=value)
            elif not self.inclusive and value <= self.min_value:
                msg = self.error_message or f"{field_name} must be > {self.min_value}"
                raise ValidationError(msg, field=field_name, value=value)

        if self.max_value is not None:
            if self.inclusive and value > self.max_value:
                msg = self.error_message or f"{field_name} must be <= {self.max_value}"
                raise ValidationError(msg, field=field_name, value=value)
            elif not self.inclusive and value >= self.max_value:
                msg = self.error_message or f"{field_name} must be < {self.max_value}"
                raise ValidationError(msg, field=field_name, value=value)

        return value


class ChoiceValidator(Validator):
    """Validate that value is one of allowed choices"""

    def __init__(self, choices: list[Any], error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.choices = choices

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if value not in self.choices:
            msg = self.error_message or f"{field_name} must be one of {self.choices}"
            raise ValidationError(msg, field=field_name, value=value)
        return value


class RegexValidator(Validator):
    """Validate that string matches regex pattern"""

    def __init__(self, pattern: str, error_message: str | None = None) -> None:
        super().__init__(error_message)
        self.pattern = re.compile(pattern)

    def validate(self, value: Any, field_name: str = "value") -> Any:
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", field=field_name, value=value)

        if not self.pattern.match(value):
            msg = self.error_message or f"{field_name} does not match required pattern"
            raise ValidationError(msg, field=field_name, value=value)

        return value
