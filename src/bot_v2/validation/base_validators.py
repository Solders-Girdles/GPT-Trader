"""
Base validator class for the validation framework.
"""

from collections.abc import Callable
from typing import Any

from bot_v2.errors import ValidationError


class Validator:
    """Base validator class.

    Provides optional predicate support so simple validation rules can be
    expressed without creating a dedicated subclass. Subclasses can override
    :meth:`validate` to provide richer behaviour.
    """

    def __init__(
        self,
        error_message: str | None = None,
        predicate: Callable[[Any], bool | tuple[bool, Any]] | None = None,
    ) -> None:
        self.error_message = error_message
        self._predicate = predicate

    def validate(self, value: Any, field_name: str = "value") -> Any:
        """Validate a value and return it if valid.

        When a predicate is supplied the validator will call it and expect a
        truthy result. Predicates may optionally return a ``(bool, value)``
        tuple to allow simple coercion. Without a predicate the validator acts
        as a no-op pass-through.
        """

        if self._predicate is None:
            return value

        try:
            result = self._predicate(value)
        except Exception as exc:  # pragma: no cover - defensive guard
            message = self.error_message or f"{field_name} failed validation"
            raise ValidationError(message, field=field_name, value=value) from exc

        transformed = value
        passed: bool

        if isinstance(result, tuple):
            if len(result) != 2:
                message = self.error_message or f"{field_name} failed validation"
                raise ValidationError(message, field=field_name, value=value)
            passed = bool(result[0])
            transformed = result[1]
        else:
            passed = bool(result)

        if not passed:
            message = self.error_message or f"{field_name} failed validation"
            raise ValidationError(message, field=field_name, value=value)

        return transformed

    def __call__(self, value: Any, field_name: str = "value") -> Any:
        return self.validate(value, field_name)
