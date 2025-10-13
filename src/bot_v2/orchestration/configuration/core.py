"""Core configuration models and defaults for GPT-Trader orchestration."""

from __future__ import annotations

import logging
from datetime import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from bot_v2.config.types import Profile
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.orchestration.symbols import (
    derivatives_enabled as _resolve_derivatives_enabled,
)
from bot_v2.orchestration.symbols import (
    normalize_symbol_list,
)

logger = logging.getLogger(__name__)

# Top spot markets we enable by default (ordered by Coinbase USD volume).
TOP_VOLUME_BASES = [
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "LTC",
    "ADA",
    "DOGE",
    "BCH",
    "AVAX",
    "LINK",
]

DEFAULT_SPOT_SYMBOLS = [f"{base}-USD" for base in TOP_VOLUME_BASES]

DEFAULT_SPOT_RISK_PATH = Path(__file__).resolve().parents[4] / "config" / "risk" / "spot_top10.json"


class BotConfig(BaseModel):
    """Bot configuration backed by Pydantic validation."""

    profile: Profile
    dry_run: bool = False
    symbols: list[str] | None = None
    derivatives_enabled: bool = False
    update_interval: int = 5
    short_ma: int = 5
    long_ma: int = 20
    target_leverage: int = 2
    trailing_stop_pct: float = 0.01
    enable_shorts: bool = False
    max_position_size: Decimal = Decimal("1000")
    max_leverage: int = 3
    reduce_only_mode: bool = False
    mock_broker: bool = False
    mock_fills: bool = False
    enable_order_preview: bool = False
    account_telemetry_interval: int = 300
    trading_window_start: time | None = None
    trading_window_end: time | None = None
    trading_days: list[str] | None = None
    daily_loss_limit: Decimal = Decimal("0")
    time_in_force: str = "GTC"
    perps_enable_streaming: bool = False
    perps_stream_level: int = 1
    perps_paper_trading: bool = False
    perps_force_mock: bool = False
    perps_position_fraction: float | None = None
    perps_skip_startup_reconcile: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict, repr=False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    @field_validator("max_leverage", mode="before")
    @classmethod
    def _validate_max_leverage(cls, value: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "max_leverage_invalid",
                "max_leverage must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "max_leverage_too_small",
                "max_leverage must be positive, got {value}",
                {"value": result},
            )
        return result

    @field_validator("update_interval", mode="before")
    @classmethod
    def _validate_update_interval(cls, value: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "update_interval_invalid",
                "update_interval must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "update_interval_too_small",
                "update_interval must be positive, got {value}",
                {"value": result},
            )
        return result

    @field_validator("max_position_size", mode="before")
    @classmethod
    def _validate_max_position_size(cls, value: Any) -> Decimal:
        try:
            result = Decimal(str(value))
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise PydanticCustomError(
                "max_position_size_invalid",
                "max_position_size must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "max_position_size_too_small",
                "max_position_size must be positive, got {value}",
                {"value": str(result)},
            )
        return result

    @field_validator("daily_loss_limit", mode="before")
    @classmethod
    def _validate_daily_loss_limit(cls, value: Any) -> Decimal:
        try:
            result = Decimal(str(value))
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise PydanticCustomError(
                "daily_loss_limit_invalid",
                "daily_loss_limit must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result < 0:
            raise PydanticCustomError(
                "daily_loss_limit_negative",
                "daily_loss_limit must be non-negative, got {value}",
                {"value": str(result)},
            )
        return result

    @field_validator("symbols", mode="before")
    @classmethod
    def _validate_symbols(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise PydanticCustomError(
                "symbols_invalid_type",
                "symbols must be a list or tuple, got {type}",
                {"type": type(value).__name__},
            )
        symbols = [str(item).strip().upper() for item in value]
        if not symbols:
            raise PydanticCustomError(
                "symbols_empty",
                "symbols cannot be empty when provided",
                {},
            )
        invalid = [
            f"[{idx}]: {repr(orig)}"
            for idx, (orig, item) in enumerate(zip(value, symbols))
            if not item
        ]
        if invalid:
            raise PydanticCustomError(
                "symbols_invalid_values",
                "symbols must contain only non-empty strings: {invalid}",
                {"invalid": ", ".join(invalid)},
            )
        return symbols

    @field_validator("time_in_force", mode="before")
    @classmethod
    def _validate_time_in_force(cls, value: Any) -> str | None:
        if value is None:
            return None
        tif = str(value).upper()
        supported = {"GTC", "IOC", "FOK"}
        if tif not in supported:
            raise PydanticCustomError(
                "time_in_force_unsupported",
                "time_in_force must be one of {supported}, got {value}",
                {"supported": supported, "value": repr(value)},
            )
        return tif

    @field_validator("account_telemetry_interval", mode="before")
    @classmethod
    def _validate_account_telemetry_interval(cls, value: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "account_telemetry_interval_invalid",
                "account_telemetry_interval must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                "account_telemetry_interval_too_small",
                "account_telemetry_interval must be positive, got {value}",
                {"value": result},
            )
        return result

    @field_validator("perps_stream_level", mode="before")
    @classmethod
    def _validate_perps_stream_level(cls, value: Any) -> int:
        if value is None:
            return 1
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "perps_stream_level_invalid",
                "perps_stream_level must be a valid integer, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result < 1:
            raise PydanticCustomError(
                "perps_stream_level_too_small",
                "perps_stream_level must be >= 1, got {value}",
                {"value": result},
            )
        return result

    @field_validator("perps_position_fraction", mode="before")
    @classmethod
    def _validate_perps_position_fraction(cls, value: Any) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "perps_position_fraction_invalid",
                "perps_position_fraction must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if not 0 < result <= 1:
            raise PydanticCustomError(
                "perps_position_fraction_invalid_range",
                "perps_position_fraction must be in (0, 1], got {value}",
                {"value": result},
            )
        return result

    @field_validator("target_leverage", "short_ma", "long_ma", mode="before")
    @classmethod
    def _validate_positive_integers(cls, value: Any, field: Any) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                f"{field.field_name}_invalid",
                f"{field.field_name} must be a valid integer, got {{value}}: {{error}}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result <= 0:
            raise PydanticCustomError(
                f"{field.field_name}_too_small",
                f"{field.field_name} must be positive, got {{value}}",
                {"value": result},
            )
        return result

    @field_validator("trailing_stop_pct", mode="before")
    @classmethod
    def _validate_trailing_stop_pct(cls, value: Any) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise PydanticCustomError(
                "trailing_stop_pct_invalid",
                "trailing_stop_pct must be numeric, got {value}: {error}",
                {"value": value, "error": str(exc)},
            ) from exc
        if result < 0:
            raise PydanticCustomError(
                "trailing_stop_pct_negative",
                "trailing_stop_pct must be non-negative, got {value}",
                {"value": result},
            )
        return result

    @model_validator(mode="after")
    def _apply_defaults_and_normalization(self) -> BotConfig:
        metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}

        settings = metadata.get("_runtime_settings")
        if not isinstance(settings, RuntimeSettings):
            settings = load_runtime_settings()
            metadata["_runtime_settings"] = settings
            object.__setattr__(self, "metadata", metadata)

        submitted_symbols = list(self.symbols) if self.symbols is not None else None
        last_normalized = metadata.get("normalized_symbols")
        last_normalized_list = list(last_normalized) if isinstance(last_normalized, list) else None
        requested_symbols = metadata.get("requested_symbols")
        requested_list = list(requested_symbols) if isinstance(requested_symbols, list) else None

        override_payload = metadata.get("symbol_normalization_overrides", {})
        override_quote: str | None = None
        override_derivatives: bool | None = None
        if isinstance(override_payload, dict):
            raw_quote = override_payload.get("quote")
            if raw_quote is not None:
                override_quote = str(raw_quote).upper()
            if "allow_derivatives" in override_payload:
                override_derivatives = bool(override_payload["allow_derivatives"])

        if self.symbols is None:
            object.__setattr__(self, "symbols", list(DEFAULT_SPOT_SYMBOLS))
            submitted_symbols = list(DEFAULT_SPOT_SYMBOLS)

        if submitted_symbols is not None:
            if last_normalized_list is not None and submitted_symbols != last_normalized_list:
                requested_list = list(submitted_symbols)
                metadata["requested_symbols"] = list(submitted_symbols)
            elif requested_list is None:
                requested_list = list(submitted_symbols)
                metadata["requested_symbols"] = list(submitted_symbols)

        input_symbols = requested_list if requested_list is not None else submitted_symbols

        metadata.setdefault("default_quote", settings.coinbase_default_quote)

        quote_currency = (
            override_quote
            or metadata.get("default_quote")
            or settings.coinbase_default_quote
            or "USD"
        )
        quote_currency = str(quote_currency).upper()
        if override_derivatives is None:
            allow_derivatives = _resolve_derivatives_enabled(self.profile)
        else:
            allow_derivatives = override_derivatives

        normalized, logs = normalize_symbol_list(
            input_symbols,
            allow_derivatives=allow_derivatives,
            quote=quote_currency,
        )
        object.__setattr__(self, "symbols", normalized)
        object.__setattr__(self, "derivatives_enabled", bool(allow_derivatives))

        metadata.setdefault("default_quote", quote_currency)
        metadata["symbol_normalization_logs"] = [
            {"level": record.level, "message": record.message, "args": list(record.args)}
            for record in logs
        ]
        metadata["normalized_symbols"] = list(normalized)

        if isinstance(override_payload, dict):
            existing_quote = override_payload.get("quote")
            existing_allow = override_payload.get("allow_derivatives")
        else:
            existing_quote = None
            existing_allow = None

        if (
            override_quote is not None
            or override_derivatives is not None
            or existing_quote is not None
            or existing_allow is not None
        ):
            metadata["symbol_normalization_overrides"] = {
                "quote": override_quote if override_quote is not None else existing_quote,
                "allow_derivatives": (
                    override_derivatives if override_derivatives is not None else existing_allow
                ),
            }
        elif "symbol_normalization_overrides" in metadata:
            metadata.pop("symbol_normalization_overrides", None)

        object.__setattr__(self, "metadata", metadata)
        return self

    @model_validator(mode="after")
    def _validate_profile_constraints(self) -> BotConfig:
        errors: list[str] = []

        if self.profile == Profile.CANARY:
            if not self.reduce_only_mode:
                errors.append(
                    "[canary_reduce_only_required] Canary profile must have reduce_only_mode=True"
                )
            if self.max_leverage > 1:
                errors.append(
                    f"[canary_max_leverage_too_high] Canary profile max_leverage cannot exceed 1, got {self.max_leverage}"
                )
            if self.time_in_force not in ("IOC",):
                errors.append(
                    f"[canary_time_in_force_invalid] Canary profile must use IOC time_in_force, got {self.time_in_force}"
                )

        elif self.profile == Profile.SPOT:
            if self.enable_shorts:
                errors.append("[spot_shorts_not_allowed] Spot profile cannot enable shorts")
            if self.max_leverage > 1:
                errors.append(
                    f"[spot_leverage_too_high] Spot profile max_leverage cannot exceed 1, got {self.max_leverage}"
                )

        if self.short_ma >= self.long_ma:
            errors.append(
                f"[ma_periods_invalid] short_ma ({self.short_ma}) must be < long_ma ({self.long_ma})"
            )

        if errors:
            raise ValueError("; ".join(errors))
        return self

    @classmethod
    def from_profile(
        cls, profile: str, *, settings: RuntimeSettings | None = None, **overrides: Any
    ) -> BotConfig:
        from .manager import ConfigManager  # Local import to avoid circular dependency.

        manager = ConfigManager(
            profile=profile,
            overrides=overrides,
            config_cls=cls,
            settings=settings,
        )
        return manager.build()


__all__ = [
    "BotConfig",
    "DEFAULT_SPOT_RISK_PATH",
    "DEFAULT_SPOT_SYMBOLS",
    "TOP_VOLUME_BASES",
]
