"""Collection and mapping validation rules."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from .base import BaseValidationRule, RuleError, normalize_rule


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
        self._value_rule = normalize_rule(value_rule) if value_rule else None
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
        self._item_rule = normalize_rule(item_rule) if item_rule else None
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


__all__ = ["MappingRule", "ListRule"]
