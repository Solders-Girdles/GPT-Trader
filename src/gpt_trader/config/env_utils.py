"""Shared helpers for parsing configuration values from environment variables."""

from __future__ import annotations

import os
from collections.abc import Callable
from decimal import Decimal
from typing import Any, NoReturn, TypeVar
from typing import cast as typing_cast

from gpt_trader.validation import (
    BooleanRule,
    DecimalRule,
    FloatRule,
    IntegerRule,
    ListRule,
    MappingRule,
    RuleError,
)

T = TypeVar("T")


class EnvVarError(ValueError):
    """Raised when an environment variable cannot be parsed."""

    def __init__(self, var_name: str, message: str, value: str | None = None) -> None:
        super().__init__(f"{var_name}: {message}")
        self.var_name = var_name
        self.message = message
        self.value = value


def _get_env(var_name: str) -> str | None:
    """Fetch and normalise the value of an environment variable."""
    raw = os.environ.get(var_name)
    if raw is None:
        return None
    value = raw.strip()
    return value if value else None


def _raise_env_error(var_name: str, message: str, value: str | None = None) -> NoReturn:
    raise EnvVarError(var_name, message, value)


def _normalize_rule_message(var_name: str, message: str) -> str:
    if message.startswith(var_name):
        trimmed = message[len(var_name) :]
        return trimmed.lstrip(" :")
    return message


def _raise_rule_error(var_name: str, error: RuleError) -> NoReturn:
    message = _normalize_rule_message(var_name, str(error))
    value = getattr(error, "value", None)
    _raise_env_error(var_name, message, value if isinstance(value, str) else None)


def _ensure_required(var_name: str) -> None:
    _raise_env_error(var_name, "is required but was not set")


def _coerce_with_rule(var_name: str, raw_value: str, rule: Callable[[Any, str], Any]) -> Any:
    try:
        return rule(raw_value, var_name)
    except RuleError as exc:
        _raise_rule_error(var_name, exc)


_BOOLEAN_RULE = BooleanRule()
_INT_RULE = IntegerRule()
_FLOAT_RULE = FloatRule()
_DECIMAL_RULE = DecimalRule()


def coerce_env_value(
    var_name: str,
    cast: Callable[[str], T],
    *,
    default: T | None = None,
    required: bool = False,
) -> T | None:
    """Return ``cast`` applied to ``var_name`` if it is set."""
    value = _get_env(var_name)
    if value is None:
        if required:
            _ensure_required(var_name)
        return default
    try:
        return cast(value)
    except Exception:  # pragma: no cover - defensive
        _raise_env_error(var_name, f"could not be parsed: {value!r}", value)


def require_env_value(
    var_name: str,
    cast: Callable[[str], T],
) -> T:
    """Return ``cast`` applied to ``var_name`` and raise if unset."""
    value = coerce_env_value(var_name, cast, required=True)
    if value is None:  # pragma: no cover - required=True guarantees value
        raise ValueError(f"Required environment variable {var_name} returned None after coercion")
    return value


def get_env_int(
    var_name: str,
    *,
    default: int | None = None,
    required: bool = False,
) -> int | None:
    """Read an integer from ``var_name``."""
    value = _get_env(var_name)
    if value is None:
        if required:
            _ensure_required(var_name)
        return default
    coerced = _coerce_with_rule(var_name, value, _INT_RULE)
    return int(coerced)


def get_env_float(
    var_name: str,
    *,
    default: float | None = None,
    required: bool = False,
) -> float | None:
    """Read a float from ``var_name``."""
    value = _get_env(var_name)
    if value is None:
        if required:
            _ensure_required(var_name)
        return default
    coerced = _coerce_with_rule(var_name, value, _FLOAT_RULE)
    return float(coerced)


def get_env_decimal(
    var_name: str,
    *,
    default: Decimal | None = None,
    required: bool = False,
) -> Decimal | None:
    """Read a :class:`~decimal.Decimal` from ``var_name``."""
    value = _get_env(var_name)
    if value is None:
        if required:
            _ensure_required(var_name)
        return default
    coerced = _coerce_with_rule(var_name, value, _DECIMAL_RULE)
    if not isinstance(coerced, Decimal):
        raise TypeError(f"Expected Decimal for {var_name}, got {type(coerced).__name__}")
    return coerced


def get_env_bool(
    var_name: str,
    *,
    default: bool | None = None,
    required: bool = False,
) -> bool | None:
    """Parse a boolean value from ``var_name``.

    Accepted truthy values: ``{"1", "true", "t", "yes", "y", "on"}``
    Accepted falsy values: ``{"0", "false", "f", "no", "n", "off"}``
    """

    value = _get_env(var_name)
    if value is None:
        if required:
            _ensure_required(var_name)
        return default
    return bool(_coerce_with_rule(var_name, value, _BOOLEAN_RULE))


def parse_env_list(
    var_name: str,
    cast: Callable[[str], T],
    *,
    separator: str = ",",
    allow_empty: bool = True,
    default: list[T] | None = None,
    required: bool = False,
) -> list[T]:
    """Parse a delimited list from an environment variable."""
    value = _get_env(var_name)
    if value is None:
        if required:
            _ensure_required(var_name)
        return list(default or [])

    rule = ListRule(
        item_converter=cast,
        allow_none=False,
        allow_blank_items=allow_empty,
        separator=separator,
    )
    try:
        return typing_cast(list[T], rule(value, var_name))
    except RuleError as exc:
        _raise_rule_error(var_name, exc)


def parse_env_mapping(
    var_name: str,
    cast: Callable[[str], T],
    *,
    item_delimiter: str = ",",
    kv_delimiter: str = ":",
    allow_empty: bool = True,
    default: dict[str, T] | None = None,
    required: bool = False,
) -> dict[str, T]:
    """Parse a mapping from an environment variable.

    ``var_name`` should contain values in the form ``"KEY:VALUE,KEY2:VALUE2"``.
    ``cast`` is applied to each value; errors raise :class:`EnvVarError`.
    """

    value = _get_env(var_name)
    if value is None:
        if required:
            _ensure_required(var_name)
        return dict(default or {})

    rule = MappingRule(
        value_converter=cast,
        allow_none=False,
        item_separator=item_delimiter,
        kv_separator=kv_delimiter,
        allow_blank_items=allow_empty,
    )
    try:
        return typing_cast(dict[str, T], rule(value, var_name))
    except RuleError as exc:
        _raise_rule_error(var_name, exc)


__all__ = [
    "EnvVarError",
    "coerce_env_value",
    "require_env_value",
    "get_env_bool",
    "get_env_decimal",
    "get_env_float",
    "get_env_int",
    "parse_env_list",
    "parse_env_mapping",
]
