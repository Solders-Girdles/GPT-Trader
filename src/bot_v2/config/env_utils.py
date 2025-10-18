"""Shared helpers for parsing configuration values from environment variables."""

from __future__ import annotations

from collections.abc import Callable
from decimal import Decimal
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
else:  # pragma: no cover - runtime alias for type checking
    RuntimeSettings = Any  # type: ignore[misc]

from bot_v2.validation import (
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
    runtime_settings = _resolve_settings(settings)
    value = _get_env(var_name, runtime_settings)
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
    settings: RuntimeSettings | None = None,
) -> float | None:
    """Read a float from ``var_name``."""
    runtime_settings = _resolve_settings(settings)
    value = _get_env(var_name, runtime_settings)
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
    settings: RuntimeSettings | None = None,
) -> Decimal | None:
    """Read a :class:`~decimal.Decimal` from ``var_name``."""
    runtime_settings = _resolve_settings(settings)
    value = _get_env(var_name, runtime_settings)
    if value is None:
        if required:
            _ensure_required(var_name)
        return default
    coerced = _coerce_with_rule(var_name, value, _DECIMAL_RULE)
    assert isinstance(coerced, Decimal)
    return coerced


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
    return bool(_coerce_with_rule(var_name, value, _BOOLEAN_RULE))


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

    rule = ListRule(
        item_converter=cast,
        allow_none=False,
        allow_blank_items=allow_empty,
        separator=separator,
    )
    try:
        return rule(value, var_name)
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

    rule = MappingRule(
        value_converter=cast,
        allow_none=False,
        item_separator=item_delimiter,
        kv_separator=kv_delimiter,
        allow_blank_items=allow_empty,
    )
    try:
        return rule(value, var_name)
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
