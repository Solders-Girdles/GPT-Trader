"""
Composite validators for chaining multiple validators together.
"""

from typing import Any

from .base_validators import Validator


class CompositeValidator(Validator):
    """Combine multiple validators"""

    def __init__(self, *validators: Validator) -> None:
        self.validators = validators

    def validate(self, value: Any, field_name: str = "value") -> Any:
        for validator in self.validators:
            value = validator(value, field_name)
        return value
