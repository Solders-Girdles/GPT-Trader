"""Utility helpers for configuration tracking and runtime parsing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from bot_v2.orchestration.runtime_settings import load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="config")

SLIPPAGE_ENV_KEY = "SLIPPAGE_MULTIPLIERS"

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.configuration import BotConfig


@dataclass(frozen=True)
class ConfigBaselinePayload:
    """Normalized payload describing the tracked configuration baseline."""

    data: dict[str, Any]
    fields: tuple[str, ...]

    EXCLUDE_FIELDS = frozenset({"metadata"})
    INCLUDE_FIELDS: frozenset[str] | None = None

    @classmethod
    def from_config(
        cls,
        config: BotConfig,
        *,
        derivatives_enabled: bool,
    ) -> ConfigBaselinePayload:
        """Build payload from the runtime bot configuration."""

        payload = config.model_dump(mode="python")
        payload["derivatives_enabled"] = bool(derivatives_enabled)

        include = cls.INCLUDE_FIELDS
        if include is None:
            ordered_keys = [key for key in payload if key not in cls.EXCLUDE_FIELDS]
        else:
            ordered_keys = [
                key for key in include if key in payload and key not in cls.EXCLUDE_FIELDS
            ]

        normalized: dict[str, Any] = {
            key: cls._normalize_value(payload.get(key)) for key in ordered_keys
        }

        return cls(data=normalized, fields=tuple(ordered_keys))

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to a dictionary suitable for baseline snapshots."""

        return {key: self._present_value(value) for key, value in self.data.items()}

    def diff(self, other: ConfigBaselinePayload) -> dict[str, Any]:
        """Compute field-level differences between payloads."""
        diff: dict[str, Any] = {}
        ordered_keys = dict.fromkeys(self.fields)
        for key in other.fields:
            ordered_keys.setdefault(key, None)

        for key in ordered_keys:
            left = self.data.get(key)
            right = other.data.get(key)
            if left != right:
                diff[key] = {
                    "current": self._present_value(left),
                    "new": self._present_value(right),
                }
        return diff

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        if isinstance(value, tuple):
            return tuple(value)
        return value

    @staticmethod
    def _present_value(value: Any) -> Any:
        if isinstance(value, tuple):
            return list(value)
        return value


def parse_slippage_multipliers(raw_value: str | None) -> dict[str, Decimal]:
    """Parse SLIPPAGE_MULTIPLIERS style payloads into Decimal multipliers.

    Entries must follow ``SYMBOL:MULTIPLIER`` format and be comma-separated.
    """

    multipliers: dict[str, Decimal] = {}
    if not raw_value:
        return multipliers

    entries = [entry.strip() for entry in raw_value.split(",") if entry.strip()]
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Invalid slippage entry '{entry}' (expected SYMBOL:MULTIPLIER)")

        symbol_part, value_part = entry.split(":", 1)
        symbol = symbol_part.strip()
        if not symbol:
            raise ValueError(f"Missing symbol in slippage entry '{entry}'")

        value_text = value_part.strip()
        try:
            multipliers[symbol] = Decimal(value_text)
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(f"Invalid multiplier for '{symbol}': {value_text}") from exc

    return multipliers


def load_slippage_multipliers(
    *,
    env: Mapping[str, str] | None = None,
    env_key: str = SLIPPAGE_ENV_KEY,
) -> dict[str, Decimal]:
    """Load slippage multipliers from environment, logging validation errors."""

    source = env or load_runtime_settings().raw_env
    raw_value = source.get(env_key, "")
    if not raw_value:
        return {}

    try:
        return parse_slippage_multipliers(raw_value)
    except ValueError as exc:
        logger.warning(
            "Invalid %s entry '%s': %s",
            env_key,
            raw_value,
            exc,
            operation="config_slippage_parse",
            status="invalid",
            exc_info=True,
        )
        return {}
