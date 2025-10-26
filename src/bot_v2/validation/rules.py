"""
Composable validation rules used across configuration models.

The module provides reusable building blocks for coercion and validation that
work with Pydantic field validators as well as bespoke config pipelines.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bot_v2.utilities.parsing import (
    FALSE_BOOLEAN_TOKENS,
    TRUE_BOOLEAN_TOKENS,
    interpret_tristate_bool,
)


class BaseValidationRule:
    """Base helper to make rules callable."""

    def __call__(self, value: Any, field_name: str = "value") -> Any:
        return self.apply(value, field_name=field_name)

    def apply(self, value: Any, *, field_name: str = "value") -> Any:  # pragma: no cover - abstract
        raise NotImplementedError


class RuleError(ValueError):
    """ValueError subclass that carries the offending value."""

    def __init__(self, message: str, *, value: Any | None = None) -> None:
        super().__init__(message)
        self.value = value


class FunctionRule(BaseValidationRule):
    """Adapter that converts a callable into a rule."""

    def __init__(self, func: Callable[[Any, str], Any]) -> None:
        self._func = func

    def apply(self, value: Any, *, field_name: str = "value") -> Any:
        return self._func(value, field_name)


def _normalize_rule(
    rule: BaseValidationRule | Callable[[Any, str], Any],
) -> BaseValidationRule:
    if isinstance(rule, BaseValidationRule):
        return rule
    if callable(rule):
        return FunctionRule(rule)
    raise TypeError(f"Unsupported rule type: {type(rule)!r}")


class RuleChain(BaseValidationRule):
    """Compose multiple rules executed sequentially."""

    def __init__(self, *rules: BaseValidationRule | Callable[[Any, str], Any]) -> None:
        if not rules:
            raise RuleError("RuleChain requires at least one rule")
        self._rules = tuple(_normalize_rule(rule) for rule in rules)

    def apply(self, value: Any, *, field_name: str = "value") -> Any:
        for rule in self._rules:
            value = rule(value, field_name)
        return value


class BooleanRule(BaseValidationRule):
    """Parse boolean-like values using common token conventions."""

    def __init__(
        self,
        *,
        default: bool | None = None,
        true_tokens: Iterable[str] | None = None,
        false_tokens: Iterable[str] | None = None,
    ) -> None:
        self._default = default
        self._true_tokens = {token.lower() for token in TRUE_BOOLEAN_TOKENS}
        self._false_tokens = {token.lower() for token in FALSE_BOOLEAN_TOKENS}
        if true_tokens:
            self._true_tokens.update(token.lower() for token in true_tokens)
        if false_tokens:
            self._false_tokens.update(token.lower() for token in false_tokens)
        self._token_display = ", ".join(sorted(self._true_tokens | self._false_tokens))

    def apply(self, value: Any, *, field_name: str = "value") -> bool:
        if value is None:
            if self._default is not None:
                return self._default
            raise RuleError(f"{field_name} expected a boolean but received None", value=value)

        if isinstance(value, bool):
            return value

        if isinstance(value, (int, float)):
            if value in (0, 1):
                return bool(int(value))

        if isinstance(value, str):
            candidate = interpret_tristate_bool(value.strip())
            if candidate is not None:
                return candidate
            lowered = value.strip().lower()
            if lowered in self._true_tokens:
                return True
            if lowered in self._false_tokens:
                return False

        raise RuleError(
            f"{field_name} expected a boolean value ({self._token_display}) but received {value!r}",
            value=value,
        )


class MappingRule(BaseValidationRule):
    """Parse mappings either from dict-like inputs or comma-separated strings."""

    def __init__(
        self,
        *,
        value_converter: Callable[[Any], Any] | None = None,
        value_rule: BaseValidationRule | Callable[[Any, str], Any] | None = None,
        allow_none: bool = True,
        item_separator: str = ",",
        kv_separator: str = ":",
        allow_blank_items: bool = True,
    ) -> None:
        self._value_converter = value_converter
        self._value_rule = _normalize_rule(value_rule) if value_rule else None
        self._allow_none = allow_none
        self._item_separator = item_separator
        self._kv_separator = kv_separator
        self._allow_blank_items = allow_blank_items

    def apply(self, value: Any, *, field_name: str = "value") -> dict[str, Any]:
        if value is None:
            if self._allow_none:
                return {}
            raise RuleError(f"{field_name} requires a mapping but received None", value=value)

        iterator: Iterable[tuple[Any, Any]]
        if isinstance(value, Mapping):
            iterator = value.items()
        elif isinstance(value, str):
            iterator = self._parse_string_mapping(value, field_name)
        else:
            raise RuleError(
                f"{field_name} expected a mapping or a string formatted as 'KEY{self._kv_separator}VALUE'",
                value=value,
            )

        result: dict[str, Any] = {}
        for raw_key, raw_val in iterator:
            key = str(raw_key).strip()
            if not key:
                raise RuleError(
                    f"{field_name} includes an entry with an empty key",
                    value=raw_key,
                )

            if raw_val is None:
                raise RuleError(
                    f"{field_name} includes an entry for {key!r} with an empty value",
                    value=raw_val,
                )

            value_to_use = raw_val
            if isinstance(raw_val, str):
                value_to_use = raw_val.strip()
                if not value_to_use:
                    raise RuleError(
                        f"{field_name} includes an entry for {key!r} with an empty value",
                        value=raw_val,
                    )

            if self._value_converter:
                try:
                    value_to_use = self._value_converter(value_to_use)
                except Exception as exc:  # pragma: no cover - defensive
                    raise RuleError(
                        f"{field_name} could not cast value {raw_val!r} for key {key!r}",
                        value=raw_val,
                    ) from exc

            if self._value_rule:
                value_to_use = self._value_rule(value_to_use, f"{field_name}[{key}]")

            result[key] = value_to_use

        return result

    def _parse_string_mapping(self, raw: str, field_name: str) -> list[tuple[str, str]]:
        entries: list[tuple[str, str]] = []
        for chunk in raw.split(self._item_separator):
            candidate = chunk.strip()
            if not candidate:
                if self._allow_blank_items:
                    continue
                raise RuleError(
                    f"{field_name} contains an empty mapping entry",
                    value=chunk,
                )
            if self._kv_separator not in candidate:
                raise RuleError(
                    f"{field_name} has an invalid entry {candidate!r}; expected 'KEY{self._kv_separator}VALUE'",
                    value=candidate,
                )
            key_raw, value_raw = candidate.split(self._kv_separator, 1)
            entries.append((key_raw, value_raw))
        return entries


class ListRule(BaseValidationRule):
    """Parse delimited lists or iterables."""

    def __init__(
        self,
        *,
        item_converter: Callable[[Any], Any] | None = None,
        item_rule: BaseValidationRule | Callable[[Any, str], Any] | None = None,
        allow_none: bool = True,
        allow_blank_items: bool = True,
        separator: str = ",",
    ) -> None:
        self._item_converter = item_converter
        self._item_rule = _normalize_rule(item_rule) if item_rule else None
        self._allow_none = allow_none
        self._allow_blank_items = allow_blank_items
        self._separator = separator

    def apply(self, value: Any, *, field_name: str = "value") -> list[Any]:
        if value is None:
            if self._allow_none:
                return []
            raise RuleError(f"{field_name} requires a list but received None", value=value)

        if isinstance(value, str):
            raw_items = [chunk for chunk in value.split(self._separator)]
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            raw_items = list(value)
        else:
            raise RuleError(
                f"{field_name} expected a delimited string or iterable for list parsing",
                value=value,
            )

        result: list[Any] = []
        for index, raw_item in enumerate(raw_items):
            candidate = raw_item
            if isinstance(raw_item, str):
                candidate = raw_item.strip()

            if candidate == "" or candidate is None:
                if self._allow_blank_items:
                    continue
                raise RuleError(
                    f"{field_name} contains an empty list entry at position {index}",
                    value=raw_item,
                )

            processed = candidate
            if self._item_converter is not None:
                try:
                    processed = self._item_converter(candidate)
                except Exception as exc:  # pragma: no cover - defensive
                    error_value = candidate if isinstance(candidate, str) else raw_item
                    raise RuleError(
                        f"{field_name} could not cast value {error_value!r}",
                        value=error_value,
                    ) from exc

            if self._item_rule is not None:
                processed = self._item_rule(processed, f"{field_name}[{index}]")

            result.append(processed)

        return result


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
            if 0 <= numeric_value <= 100:
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


__all__ = [
    "BaseValidationRule",
    "BooleanRule",
    "DecimalRule",
    "FloatRule",
    "IntegerRule",
    "ListRule",
    "MappingRule",
    "PercentageRule",
    "RuleChain",
    "RuleError",
    "StripStringRule",
    "TimeOfDayRule",
    "SymbolRule",
]
