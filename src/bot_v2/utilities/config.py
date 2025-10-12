"""Utility helpers for configuration tracking and runtime parsing."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

SLIPPAGE_ENV_KEY = "SLIPPAGE_MULTIPLIERS"

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.configuration import BotConfig


def _as_sequence(value: Iterable[str] | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class ConfigBaselinePayload:
    """Normalized payload describing the tracked configuration baseline."""

    profile: Any
    dry_run: bool
    symbols: tuple[str, ...]
    derivatives_enabled: bool
    update_interval: Any
    short_ma: Any
    long_ma: Any
    target_leverage: Any
    trailing_stop_pct: Any
    enable_shorts: bool
    max_position_size: Any
    max_leverage: Any
    reduce_only_mode: bool
    mock_broker: bool
    mock_fills: bool
    enable_order_preview: bool
    account_telemetry_interval: Any
    trading_window_start: Any
    trading_window_end: Any
    trading_days: tuple[str, ...]
    daily_loss_limit: Any
    time_in_force: Any
    perps_enable_streaming: bool
    perps_stream_level: Any
    perps_paper_trading: bool
    perps_force_mock: bool
    perps_position_fraction: Any
    perps_skip_startup_reconcile: bool

    @classmethod
    def from_config(
        cls,
        config: BotConfig,
        *,
        derivatives_enabled: bool,
    ) -> ConfigBaselinePayload:
        """Build payload from the runtime bot configuration."""

        return cls(
            profile=getattr(config, "profile", None),
            dry_run=bool(getattr(config, "dry_run", False)),
            symbols=_as_sequence(getattr(config, "symbols", None)),
            derivatives_enabled=bool(derivatives_enabled),
            update_interval=getattr(config, "update_interval", None),
            short_ma=getattr(config, "short_ma", None),
            long_ma=getattr(config, "long_ma", None),
            target_leverage=getattr(config, "target_leverage", None),
            trailing_stop_pct=getattr(config, "trailing_stop_pct", None),
            enable_shorts=bool(getattr(config, "enable_shorts", False)),
            max_position_size=getattr(config, "max_position_size", None),
            max_leverage=getattr(config, "max_leverage", None),
            reduce_only_mode=bool(getattr(config, "reduce_only_mode", False)),
            mock_broker=bool(getattr(config, "mock_broker", False)),
            mock_fills=bool(getattr(config, "mock_fills", False)),
            enable_order_preview=bool(getattr(config, "enable_order_preview", False)),
            account_telemetry_interval=getattr(config, "account_telemetry_interval", None),
            trading_window_start=getattr(config, "trading_window_start", None),
            trading_window_end=getattr(config, "trading_window_end", None),
            trading_days=_as_sequence(getattr(config, "trading_days", None)),
            daily_loss_limit=getattr(config, "daily_loss_limit", None),
            time_in_force=getattr(config, "time_in_force", None),
            perps_enable_streaming=bool(getattr(config, "perps_enable_streaming", False)),
            perps_stream_level=getattr(config, "perps_stream_level", None),
            perps_paper_trading=bool(getattr(config, "perps_paper_trading", False)),
            perps_force_mock=bool(getattr(config, "perps_force_mock", False)),
            perps_position_fraction=getattr(config, "perps_position_fraction", None),
            perps_skip_startup_reconcile=bool(
                getattr(config, "perps_skip_startup_reconcile", False)
            ),
        )

    @classmethod
    def tracked_fields(cls) -> tuple[str, ...]:
        """Return the ordered field names tracked for drift detection."""
        return tuple(field.name for field in fields(cls))

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to a dictionary suitable for baseline snapshots."""

        def _normalize(value: Any) -> Any:
            if isinstance(value, tuple):
                return list(value)
            return value

        return {field.name: _normalize(getattr(self, field.name)) for field in fields(self)}

    def diff(self, other: ConfigBaselinePayload) -> dict[str, Any]:
        """Compute field-level differences between payloads."""
        diff: dict[str, Any] = {}
        for field in fields(self):
            left = getattr(self, field.name)
            right = getattr(other, field.name)
            if left != right:
                diff[field.name] = {
                    "current": list(left) if isinstance(left, tuple) else left,
                    "new": list(right) if isinstance(right, tuple) else right,
                }
        return diff


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

    source = env or os.environ
    raw_value = source.get(env_key, "")
    if not raw_value:
        return {}

    try:
        return parse_slippage_multipliers(raw_value)
    except ValueError as exc:
        logger.warning("Invalid %s entry '%s': %s", env_key, raw_value, exc, exc_info=True)
        return {}
