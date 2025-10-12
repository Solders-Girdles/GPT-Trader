"""Configuration parsing utilities for consistent environment variable handling."""

from __future__ import annotations

import os
from collections.abc import Callable
from decimal import Decimal
from typing import TypeVar

from bot_v2.config.env_utils import (
    EnvVarError,
    get_env_bool,
    get_env_decimal,
    get_env_float,
    get_env_int,
    parse_env_list,
    parse_env_mapping,
)
from bot_v2.errors import ValidationError

T = TypeVar("T")


def _wrap_env_error(var_name: str, error: EnvVarError) -> ValidationError:
    return ValidationError(
        f"Failed to parse {var_name}: {error.message}",
        field=error.var_name,
        value=error.value,
    )


def parse_mapping_env(
    var_name: str,
    cast: Callable[[str], T],
    default: dict[str, T] | None = None,
    separator: str = ":",
    kv_delimiter: str = "=",
) -> dict[str, T]:
    """Parse a delimited environment variable into a mapping."""
    try:
        return parse_env_mapping(
            var_name,
            cast,
            item_delimiter=separator,
            kv_delimiter=kv_delimiter,
            default=default,
        )
    except EnvVarError as exc:
        raise _wrap_env_error(var_name, exc) from exc


def parse_list_env(
    var_name: str,
    cast: Callable[[str], T],
    default: list[T] | None = None,
    separator: str = ",",
) -> list[T]:
    """Parse a delimited environment variable into a list."""
    try:
        return parse_env_list(
            var_name,
            cast,
            separator=separator,
            default=default,
        )
    except EnvVarError as exc:
        raise _wrap_env_error(var_name, exc) from exc


def parse_bool_env(var_name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""
    try:
        value = get_env_bool(var_name, default=default)
    except EnvVarError as exc:
        raise _wrap_env_error(var_name, exc) from exc
    return default if value is None else value


def parse_decimal_env(var_name: str, default: Decimal | None = None) -> Decimal | None:
    """Parse a decimal environment variable."""
    try:
        return get_env_decimal(var_name, default=default)
    except EnvVarError as exc:
        raise _wrap_env_error(var_name, exc) from exc


def parse_int_env(var_name: str, default: int | None = None) -> int | None:
    """Parse an integer environment variable."""
    try:
        return get_env_int(var_name, default=default)
    except EnvVarError as exc:
        raise _wrap_env_error(var_name, exc) from exc


def parse_float_env(var_name: str, default: float | None = None) -> float | None:
    """Parse a float environment variable."""
    try:
        return get_env_float(var_name, default=default)
    except EnvVarError as exc:
        raise _wrap_env_error(var_name, exc) from exc


def validate_required_env(var_names: list[str]) -> None:
    """Validate that required environment variables are present."""
    missing = [name for name in var_names if not os.getenv(name)]
    if missing:
        raise ValidationError(f"Missing required environment variables: {', '.join(missing)}")
