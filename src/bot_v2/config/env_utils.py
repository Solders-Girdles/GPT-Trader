"""Shared helpers for parsing configuration values from environment variables."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
else:  # pragma: no cover - runtime alias for type checking
    RuntimeSettings = Any  # type: ignore[misc]

from bot_v2.utilities.parsing import (
    FALSE_BOOLEAN_TOKENS,
    TRUE_BOOLEAN_TOKENS,
    interpret_tristate_bool,
)

T = TypeVar("T")


class EnvVarError(ValueError):
    """Raised when an environment variable cannot be parsed."""

    def __init__(self, var_name: str, message: str, value: str | None = None) -> None:
        super().__init__(f"{var_name}: {message}")
        self.var_name = var_name
        self.message = message
        self.value = value


def _load_runtime_settings() -> RuntimeSettings:
    from bot_v2.orchestration.runtime_settings import load_runtime_settings as _loader

    return _loader()


def _resolve_settings(settings: RuntimeSettings | None) -> RuntimeSettings:
    return settings if settings is not None else _load_runtime_settings()


def _get_env(var_name: str, runtime_settings: RuntimeSettings) -> str | None:
    """Fetch and normalise the value of an environment variable."""
    raw = runtime_settings.raw_env.get(var_name)
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
    settings: RuntimeSettings | None = None,
) -> T | None:
    """Return ``cast`` applied to ``var_name`` if it is set."""
    runtime_settings = _resolve_settings(settings)
    value = _get_env(var_name, runtime_settings)
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
    *,
    settings: RuntimeSettings | None = None,
) -> T:
    """Return ``cast`` applied to ``var_name`` and raise if unset."""
    value = coerce_env_value(var_name, cast, required=True, settings=settings)
    assert value is not None  # pragma: no cover - required=True guarantees value
    return value


def get_env_int(
    var_name: str,
    *,
    default: int | None = None,
    required: bool = False,
    settings: RuntimeSettings | None = None,
) -> int | None:
    """Read an integer from ``var_name``."""
    return coerce_env_value(
        var_name,
        int,
        default=default,
        required=required,
        settings=settings,
    )


def get_env_float(
    var_name: str,
    *,
    default: float | None = None,
    required: bool = False,
    settings: RuntimeSettings | None = None,
) -> float | None:
    """Read a float from ``var_name``."""
    return coerce_env_value(
        var_name,
        float,
        default=default,
        required=required,
        settings=settings,
    )


def get_env_decimal(
    var_name: str,
    *,
    default: Decimal | None = None,
    required: bool = False,
    settings: RuntimeSettings | None = None,
) -> Decimal | None:
    """Read a :class:`~decimal.Decimal` from ``var_name``."""
    return coerce_env_value(
        var_name,
        Decimal,
        default=default,
        required=required,
        settings=settings,
    )


def get_env_bool(
    var_name: str,
    *,
    default: bool | None = None,
    required: bool = False,
    settings: RuntimeSettings | None = None,
) -> bool | None:
    """Parse a boolean value from ``var_name``.

    Accepted truthy values: ``{"1", "true", "t", "yes", "y", "on"}``
    Accepted falsy values: ``{"0", "false", "f", "no", "n", "off"}``
    """

    runtime_settings = _resolve_settings(settings)
    value = _get_env(var_name, runtime_settings)
    if value is None:
        if required:
            _ensure_required(var_name)
        return default

    interpreted = interpret_tristate_bool(value)
    if interpreted is not None:
        return interpreted

    allowed_values = ", ".join(sorted(TRUE_BOOLEAN_TOKENS | FALSE_BOOLEAN_TOKENS))
    _raise_env_error(
        var_name,
        f"expected a boolean value ({allowed_values}) but received {value!r}",
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
    settings: RuntimeSettings | None = None,
) -> list[T]:
    """Parse a delimited list from an environment variable."""
    runtime_settings = _resolve_settings(settings)
    value = _get_env(var_name, runtime_settings)
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
    settings: RuntimeSettings | None = None,
) -> dict[str, T]:
    """Parse a mapping from an environment variable.

    ``var_name`` should contain values in the form ``"KEY:VALUE,KEY2:VALUE2"``.
    ``cast`` is applied to each value; errors raise :class:`EnvVarError`.
    """

    runtime_settings = _resolve_settings(settings)
    value = _get_env(var_name, runtime_settings)
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
