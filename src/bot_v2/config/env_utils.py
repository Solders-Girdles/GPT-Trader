"""Shared helpers for parsing configuration values from environment variables."""

from __future__ import annotations

import os
from decimal import Decimal
from typing import Callable, Dict, Optional, TypeVar

T = TypeVar("T")

_TRUE_VALUES = {"1", "true", "t", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "f", "no", "n", "off"}


class EnvVarError(ValueError):
    """Raised when an environment variable cannot be parsed."""


def _get_env(var_name: str) -> Optional[str]:
    """Fetch and normalise the value of an environment variable."""
    raw = os.getenv(var_name)
    if raw is None:
        return None
    value = raw.strip()
    return value if value else None


def coerce_env_value(var_name: str, cast: Callable[[str], T]) -> Optional[T]:
    """Return ``cast`` applied to ``var_name`` if it is set."""
    value = _get_env(var_name)
    if value is None:
        return None
    try:
        return cast(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise EnvVarError(
            f"Environment variable {var_name} could not be parsed: {value!r}"
        ) from exc


def get_env_int(var_name: str) -> Optional[int]:
    """Read an integer from ``var_name`` if present."""
    return coerce_env_value(var_name, int)


def get_env_float(var_name: str) -> Optional[float]:
    """Read a float from ``var_name`` if present."""
    return coerce_env_value(var_name, float)


def get_env_decimal(var_name: str) -> Optional[Decimal]:
    """Read a :class:`~decimal.Decimal` from ``var_name`` if present."""
    return coerce_env_value(var_name, Decimal)


def get_env_bool(var_name: str) -> Optional[bool]:
    """Parse a boolean value from ``var_name``.

    Accepted truthy values: ``{"1", "true", "t", "yes", "y", "on"}``
    Accepted falsy values: ``{"0", "false", "f", "no", "n", "off"}``
    """

    value = _get_env(var_name)
    if value is None:
        return None

    normalised = value.lower()
    if normalised in _TRUE_VALUES:
        return True
    if normalised in _FALSE_VALUES:
        return False

    raise EnvVarError(
        f"Environment variable {var_name} expected a boolean value but received {value!r}."
    )


def parse_env_mapping(
    var_name: str,
    cast: Callable[[str], T],
    *,
    item_delimiter: str = ",",
    kv_delimiter: str = ":",
    allow_empty: bool = True,
) -> Dict[str, T]:
    """Parse a mapping from an environment variable.

    ``var_name`` should contain values in the form ``"KEY:VALUE,KEY2:VALUE2"``.
    ``cast`` is applied to each value; errors raise :class:`EnvVarError`.
    """

    value = _get_env(var_name)
    if value is None:
        return {}

    result: Dict[str, T] = {}
    pairs = value.split(item_delimiter)
    for raw_pair in pairs:
        pair = raw_pair.strip()
        if not pair:
            if allow_empty:
                continue
            raise EnvVarError(
                f"Environment variable {var_name} contains an empty mapping entry."
            )

        if kv_delimiter not in pair:
            raise EnvVarError(
                f"Environment variable {var_name} has an invalid entry {pair!r}; expected 'KEY{kv_delimiter}VALUE'."
            )

        key, raw_val = pair.split(kv_delimiter, 1)
        key = key.strip()
        if not key:
            raise EnvVarError(
                f"Environment variable {var_name} includes an entry with an empty key."
            )

        raw_val = raw_val.strip()
        if not raw_val:
            raise EnvVarError(
                f"Environment variable {var_name} includes an entry for {key!r} with an empty value."
            )

        try:
            result[key] = cast(raw_val)
        except Exception as exc:
            raise EnvVarError(
                f"Environment variable {var_name} could not cast value {raw_val!r} for key {key!r}."
            ) from exc

    return result


__all__ = [
    "EnvVarError",
    "coerce_env_value",
    "get_env_bool",
    "get_env_decimal",
    "get_env_float",
    "get_env_int",
    "parse_env_mapping",
]
