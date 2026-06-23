"""Numeric validation rules."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .base import BaseValidationRule, RuleError


class IntegerRule(BaseValidationRule):
    """Coerce values to integers."""

    def __init__(self, *, allow_none: bool = False) -> None:
        self._allow_none = allow_none

    def apply(self, value: Any, *, field_name: str = "value") -> int | None:
        if value is None:
            if self._allow_none:
                return None
            raise RuleError(f"{field_name} expected an integer but received None", value=value)
        try:
            return int(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuleError(
                f"{field_name} expected an integer-compatible value but received {value!r}",
                value=value,
            ) from exc


class DecimalRule(BaseValidationRule):
    """Coerce inputs into ``Decimal`` instances."""

    def __init__(
        self,
        *,
        default: Decimal | None = None,
        allow_none: bool = False,
        strip_strings: bool = True,
    ) -> None:
        self._default = default
        self._allow_none = allow_none
        self._strip_strings = strip_strings

    def apply(self, value: Any, *, field_name: str = "value") -> Decimal | None:
        if value is None:
            if self._default is not None:
                return self._default
            if self._allow_none:
                return None
            raise RuleError(
                f"{field_name} expected a decimal-compatible value but received None",
                value=value,
            )

        if isinstance(value, Decimal):
            return value

        if isinstance(value, (int, float)):
            return Decimal(str(value))

        if isinstance(value, str):
            candidate = value.strip() if self._strip_strings else value
            if not candidate:
                if self._default is not None:
                    return self._default
                if self._allow_none:
                    return None
                raise RuleError(
                    f"{field_name} expected a decimal value but received an empty string",
                    value=value,
                )
            try:
                return Decimal(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuleError(
                    f"{field_name} could not parse decimal value {value!r}",
                    value=value,
                ) from exc

        raise RuleError(
            f"{field_name} expected a decimal-compatible value, received {type(value).__name__}",
            value=value,
        )


class FloatRule(BaseValidationRule):
    """Coerce numeric inputs to floats with optional defaults."""

    def __init__(
        self,
        *,
        default: float | None = None,
        allow_none: bool = False,
        treat_blank_as_none: bool = True,
    ) -> None:
        self._default = default
        self._allow_none = allow_none
        self._treat_blank_as_none = treat_blank_as_none

    def apply(self, value: Any, *, field_name: str = "value") -> float | None:
        if value is None:
            if self._default is not None:
                return self._default
            if self._allow_none:
                return None
            raise RuleError(
                f"{field_name} expected a float-compatible value but received None",
                value=value,
            )

        if isinstance(value, str):
            candidate = value.strip() if self._treat_blank_as_none else value
            if not candidate and self._treat_blank_as_none:
                if self._default is not None:
                    return self._default
                if self._allow_none:
                    return None
                raise RuleError(
                    f"{field_name} expected a float value but received an empty string",
                    value=value,
                )
            try:
                return float(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                raise RuleError(
                    f"{field_name} expected a float value but received {value!r}",
                    value=value,
                ) from exc

        try:
            return float(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuleError(
                f"{field_name} expected a float value but received {value!r}",
                value=value,
            ) from exc


class PercentageRule(BaseValidationRule):
    """Ensure values fall within the inclusive [0, 1] range."""

    def __init__(self, *, allow_none: bool = False) -> None:
        self._allow_none = allow_none

    def apply(self, value: Any, *, field_name: str = "value") -> float | None:
        if value is None:
            if self._allow_none:
                return None
            raise RuleError(f"{field_name} expected a percentage but received None", value=value)

        try:
            numeric_value = float(value)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuleError(
                f"{field_name} expected a percentage value but received {value!r}",
                value=value,
            ) from exc

        if numeric_value > 1:
            if 0 <= numeric_value <= 100 and float(numeric_value).is_integer():
                numeric_value = numeric_value / 100
            else:
                raise RuleError(
                    f"{field_name} must be between 0 and 1 inclusive",
                    value=value,
                )

        if not 0 <= numeric_value <= 1:
            raise RuleError(
                f"{field_name} must be between 0 and 1 inclusive",
                value=value,
            )

        return numeric_value


__all__ = ["IntegerRule", "DecimalRule", "FloatRule", "PercentageRule"]
