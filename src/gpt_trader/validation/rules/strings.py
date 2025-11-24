"""String-oriented validation rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .base import BaseValidationRule, RuleError


class TimeOfDayRule(BaseValidationRule):
    """Validate HH:MM (24h) formatted strings."""

    _PATTERN = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")

    def __init__(self, *, allow_none: bool = True) -> None:
        self._allow_none = allow_none

    def apply(self, value: Any, *, field_name: str = "value") -> str | None:
        if value is None:
            if self._allow_none:
                return None
            raise RuleError(f"{field_name} expected time as HH:MM but received None", value=value)

        if not isinstance(value, str):
            raise RuleError(
                f"{field_name} expected time as HH:MM but received {type(value).__name__}",
                value=value,
            )

        candidate = value.strip()
        if not candidate:
            if self._allow_none:
                return None
            raise RuleError(
                f"{field_name} expected time as HH:MM but received an empty string",
                value=value,
            )

        if not self._PATTERN.match(candidate):
            raise RuleError(f"{field_name} expected HH:MM (24h) format", value=value)

        return candidate


@dataclass(frozen=True)
class StripStringRule(BaseValidationRule):
    """Normalize strings and apply a fallback default when empty."""

    default: str | None = None

    def apply(self, value: Any, *, field_name: str = "value") -> str:
        if value is None:
            if self.default is not None:
                return self.default
            raise RuleError(f"{field_name} expected a string but received None", value=value)

        candidate = str(value).strip()
        if not candidate:
            if self.default is not None:
                return self.default
            raise RuleError(f"{field_name} expected a non-empty string", value=value)

        return candidate


class SymbolRule(BaseValidationRule):
    """Validate and normalise trading symbols."""

    _PATTERN = re.compile(r"^[A-Z0-9]{1,10}(?:-[A-Z0-9]{2,10})?$")

    def __init__(self, *, uppercase: bool = True) -> None:
        self._uppercase = uppercase

    def apply(self, value: Any, *, field_name: str = "value") -> str:
        if not isinstance(value, str):
            raise RuleError(
                f"{field_name} expected a string but received {type(value).__name__}", value=value
            )

        candidate = value.strip()
        if not candidate:
            raise RuleError(f"{field_name} expected a non-empty string", value=value)

        normalised = candidate.upper() if self._uppercase else candidate
        if not self._PATTERN.match(normalised):
            raise RuleError(
                f"{field_name} must be a valid symbol (e.g., BTC-USD, ETH-PERP)",
                value=value,
            )

        return normalised


__all__ = ["StripStringRule", "SymbolRule", "TimeOfDayRule"]
