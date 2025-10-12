"""Shared helpers for parsing configuration values from environment variables."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import NoReturn, TypeVar

T = TypeVar("T")

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


class EnvVarError(ValueError):
    """Raised when an environment variable cannot be parsed."""

    def __init__(self, var_name: str, message: str, value: str | None = None) -> None:
        super().__init__(f"{var_name}: {message}")
        self.var_name = var_name
        self.message = message
        self.value = value


def _get_env(var_name: str) -> str | None:
    """Fetch and normalise the value of an environment variable."""
    raw = os.getenv(var_name)
    if raw is None:
        return None
    value = raw.strip()
    return value if value else None


def _raise_env_error(var_name: str, message: str, value: str | None = None) -> NoReturn:
    raise EnvVarError(var_name, message, value)


def _ensure_required(var_name: str) -> None:
    _raise_env_error(var_name, "is required but was not set")


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


def require_env_value(var_name: str, cast: Callable[[str], T]) -> T:
    """Return ``cast`` applied to ``var_name`` and raise if unset."""
    value = coerce_env_value(var_name, cast, required=True)
    assert value is not None  # pragma: no cover - required=True guarantees value
    return value


def get_env_int(var_name: str, *, default: int | None = None, required: bool = False) -> int | None:
    """Read an integer from ``var_name``."""
    return coerce_env_value(var_name, int, default=default, required=required)


def get_env_float(
    var_name: str, *, default: float | None = None, required: bool = False
) -> float | None:
    """Read a float from ``var_name``."""
    return coerce_env_value(var_name, float, default=default, required=required)


def get_env_decimal(
    var_name: str, *, default: Decimal | None = None, required: bool = False
) -> Decimal | None:
    """Read a :class:`~decimal.Decimal` from ``var_name``."""
    return coerce_env_value(var_name, Decimal, default=default, required=required)


def get_env_bool(
    var_name: str, *, default: bool | None = None, required: bool = False
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

    normalised = value.lower()
    if normalised in _TRUE_VALUES:
        return True
    if normalised in _FALSE_VALUES:
        return False

    _raise_env_error(
        var_name,
        f"expected a boolean value but received {value!r}",
        value,
    )
    return None  # pragma: no cover - raise above


def _split_items(value: str, delimiter: str) -> Iterable[str]:
    return (item.strip() for item in value.split(delimiter))


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

    result: list[T] = []
    for raw in _split_items(value, separator):
        if not raw:
            if allow_empty:
                continue
            _raise_env_error(var_name, "contains an empty list entry")
        try:
            result.append(cast(raw))
        except Exception:
            _raise_env_error(var_name, f"could not cast value {raw!r}", raw)
    return result


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

    result: dict[str, T] = {}
    for raw_pair in _split_items(value, item_delimiter):
        if not raw_pair:
            if allow_empty:
                continue
            _raise_env_error(var_name, "contains an empty mapping entry", raw_pair)

        if kv_delimiter not in raw_pair:
            _raise_env_error(
                var_name,
                f"has an invalid entry {raw_pair!r}; expected 'KEY{kv_delimiter}VALUE'",
                raw_pair,
            )

        key, raw_val = raw_pair.split(kv_delimiter, 1)
        key = key.strip()
        if not key:
            _raise_env_error(var_name, "includes an entry with an empty key", raw_pair)

        raw_val = raw_val.strip()
        if not raw_val:
            _raise_env_error(
                var_name,
                f"includes an entry for {key!r} with an empty value",
                raw_pair,
            )

        try:
            result[key] = cast(raw_val)
        except Exception:
            _raise_env_error(
                var_name,
                f"could not cast value {raw_val!r} for key {key!r}",
                raw_val,
            )

    return result


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
