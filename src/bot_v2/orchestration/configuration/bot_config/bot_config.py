"""Pydantic BotConfig definition with field validations and defaults."""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from bot_v2.config.types import Profile
from bot_v2.orchestration.configuration.bot_config.defaults import DEFAULT_SPOT_SYMBOLS
from bot_v2.orchestration.configuration.bot_config.rules import (
    DECIMAL_RULE as _DECIMAL_RULE,
    FLOAT_RULE as _FLOAT_RULE,
    INT_RULE as _INT_RULE,
    STRING_RULE as _STRING_RULE,
    SYMBOL_LIST_RULE as _SYMBOL_LIST_RULE,
    apply_rule as _apply_rule,
    ensure_condition as _ensure_condition,
)
from bot_v2.orchestration.configuration.bot_config.state import ConfigState
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.orchestration.symbols import (
    derivatives_enabled as _resolve_derivatives_enabled,
    normalize_symbol_list,
)


class BotConfig(BaseModel):
    """Bot configuration backed by Pydantic validation."""

    profile: Profile
    dry_run: bool = False
    symbols: list[str] | None = None
    derivatives_enabled: bool = False
    us_futures_enabled: bool = False
    intx_perpetuals_enabled: bool = False
    adx_filter_enabled: bool = False
    adx_period: int = 14
    adx_threshold: float = 25.0
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
    state: ConfigState = Field(default_factory=ConfigState, repr=False, exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    @field_validator("max_leverage", mode="before")
    @classmethod
    def _validate_max_leverage(cls, value: Any, info: FieldValidationInfo) -> int:
        result = _apply_rule(
            _INT_RULE,
            value,
            field_label="max_leverage",
            error_code="max_leverage_invalid",
            error_template="max_leverage must be a valid integer, got {value}: {error}",
        )
        profile = info.data.get("profile")
        profile_value = (
            profile.value
            if isinstance(profile, Profile)
            else str(profile) if profile is not None else "unknown"
        )
        _ensure_condition(
            result <= 0,
            error_code="max_leverage_too_small",
            error_template="max_leverage must be positive for profile {profile}, got {value}",
            context={
                "value": result,
                "profile": profile_value,
                "field": "max_leverage",
            },
        )
        return int(result)

    @field_validator("update_interval", mode="before")
    @classmethod
    def _validate_update_interval(cls, value: Any) -> int:
        result = _apply_rule(
            _INT_RULE,
            value,
            field_label="update_interval",
            error_code="update_interval_invalid",
            error_template="update_interval must be a valid integer, got {value}: {error}",
        )
        _ensure_condition(
            result <= 0,
            error_code="update_interval_too_small",
            error_template="update_interval must be positive, got {value}",
            context={"value": result},
        )
        return int(result)

    @field_validator("max_position_size", mode="before")
    @classmethod
    def _validate_max_position_size(cls, value: Any) -> Decimal:
        result = cast(
            Decimal,
            _apply_rule(
                _DECIMAL_RULE,
                value,
                field_label="max_position_size",
                error_code="max_position_size_invalid",
                error_template="max_position_size must be numeric, got {value}: {error}",
            ),
        )
        _ensure_condition(
            result <= 0,
            error_code="max_position_size_too_small",
            error_template="max_position_size must be positive, got {value}",
            context={"value": str(result)},
        )
        return result

    @field_validator("daily_loss_limit", mode="before")
    @classmethod
    def _validate_daily_loss_limit(cls, value: Any) -> Decimal:
        result = cast(
            Decimal,
            _apply_rule(
                _DECIMAL_RULE,
                value,
                field_label="daily_loss_limit",
                error_code="daily_loss_limit_invalid",
                error_template="daily_loss_limit must be numeric, got {value}: {error}",
            ),
        )
        _ensure_condition(
            result < 0,
            error_code="daily_loss_limit_negative",
            error_template="daily_loss_limit must be non-negative, got {value}",
            context={"value": str(result)},
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
        symbols = cast(
            list[str],
            _apply_rule(
                _SYMBOL_LIST_RULE,
                value,
                field_label="symbols",
                error_code="symbols_invalid_values",
                error_template="symbols must contain only non-empty strings: {error}",
            ),
        )

        if not symbols:
            raise PydanticCustomError(
                "symbols_empty",
                "symbols cannot be empty when provided",
                {},
            )
        return symbols

    @field_validator("time_in_force", mode="before")
    @classmethod
    def _validate_time_in_force(cls, value: Any) -> str | None:
        if value is None:
            return None
        raw = cast(
            str,
            _apply_rule(
                _STRING_RULE,
                value,
                field_label="time_in_force",
                error_code="time_in_force_invalid",
                error_template="time_in_force must be a non-empty string, got {value}: {error}",
            ),
        )
        tif = raw.upper()
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
        result = _apply_rule(
            _INT_RULE,
            value,
            field_label="account_telemetry_interval",
            error_code="account_telemetry_interval_invalid",
            error_template="account_telemetry_interval must be a valid integer, got {value}: {error}",
        )
        _ensure_condition(
            result <= 0,
            error_code="account_telemetry_interval_too_small",
            error_template="account_telemetry_interval must be positive, got {value}",
            context={"value": result},
        )
        return int(result)

    @field_validator("perps_stream_level", mode="before")
    @classmethod
    def _validate_perps_stream_level(cls, value: Any) -> int:
        if value is None:
            return 1
        result = _apply_rule(
            _INT_RULE,
            value,
            field_label="perps_stream_level",
            error_code="perps_stream_level_invalid",
            error_template="perps_stream_level must be a valid integer, got {value}: {error}",
        )
        _ensure_condition(
            result < 1,
            error_code="perps_stream_level_too_small",
            error_template="perps_stream_level must be >= 1, got {value}",
            context={"value": result},
        )
        return int(result)

    @field_validator("perps_position_fraction", mode="before")
    @classmethod
    def _validate_perps_position_fraction(cls, value: Any) -> float | None:
        if value is None:
            return None
        result = _apply_rule(
            _FLOAT_RULE,
            value,
            field_label="perps_position_fraction",
            error_code="perps_position_fraction_invalid",
            error_template="perps_position_fraction must be numeric, got {value}: {error}",
        )
        _ensure_condition(
            not 0 < result <= 1,
            error_code="perps_position_fraction_invalid_range",
            error_template="perps_position_fraction must be in (0, 1], got {value}",
            context={"value": result},
        )
        return float(result)

    @field_validator("target_leverage", "short_ma", "long_ma", mode="before")
    @classmethod
    def _validate_positive_integers(cls, value: Any, field: Any) -> int:
        field_name = field.field_name
        result = _apply_rule(
            _INT_RULE,
            value,
            field_label=field_name,
            error_code=f"{field_name}_invalid",
            error_template=f"{field_name} must be a valid integer, got {{value}}: {{error}}",
        )
        _ensure_condition(
            result <= 0,
            error_code=f"{field_name}_too_small",
            error_template=f"{field_name} must be positive, got {{value}}",
            context={"value": result},
        )
        return int(result)

    @field_validator("adx_period", mode="before")
    @classmethod
    def _validate_adx_period(cls, value: Any) -> int:
        result = _apply_rule(
            _INT_RULE,
            value,
            field_label="adx_period",
            error_code="adx_period_invalid",
            error_template="adx_period must be a valid integer, got {value}: {error}",
        )
        _ensure_condition(
            result < 7,
            error_code="adx_period_too_small",
            error_template="adx_period must be at least 7, got {value}",
            context={"value": result},
        )
        _ensure_condition(
            result > 50,
            error_code="adx_period_too_large",
            error_template="adx_period must be <= 50, got {value}",
            context={"value": result},
        )
        return int(result)

    @field_validator("adx_threshold", mode="before")
    @classmethod
    def _validate_adx_threshold(cls, value: Any) -> float:
        result = _apply_rule(
            _FLOAT_RULE,
            value,
            field_label="adx_threshold",
            error_code="adx_threshold_invalid",
            error_template="adx_threshold must be numeric, got {value}: {error}",
        )
        _ensure_condition(
            result < 0,
            error_code="adx_threshold_negative",
            error_template="adx_threshold must be non-negative, got {value}",
            context={"value": result},
        )
        _ensure_condition(
            result > 100,
            error_code="adx_threshold_too_high",
            error_template="adx_threshold must be <= 100, got {value}",
            context={"value": result},
        )
        return float(result)

    @field_validator("trailing_stop_pct", mode="before")
    @classmethod
    def _validate_trailing_stop_pct(cls, value: Any) -> float:
        result = _apply_rule(
            _FLOAT_RULE,
            value,
            field_label="trailing_stop_pct",
            error_code="trailing_stop_pct_invalid",
            error_template="trailing_stop_pct must be numeric, got {value}: {error}",
        )
        _ensure_condition(
            result < 0,
            error_code="trailing_stop_pct_negative",
            error_template="trailing_stop_pct must be non-negative, got {value}",
            context={"value": result},
        )
        return float(result)

    @model_validator(mode="after")
    def _apply_defaults_and_normalization(self) -> BotConfig:
        metadata = dict(self.metadata) if isinstance(self.metadata, dict) else {}

        if not isinstance(self.state, ConfigState):
            object.__setattr__(self, "state", ConfigState())

        settings = self.state.runtime_settings
        if not isinstance(settings, RuntimeSettings):
            settings = metadata.get("_runtime_settings")
            if not isinstance(settings, RuntimeSettings):
                settings = load_runtime_settings()
            self.state.runtime_settings = settings
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
        from ..manager import ConfigManager  # Local import to avoid circular dependency.

        manager = ConfigManager(
            profile=profile,
            overrides=overrides,
            config_cls=cls,
            settings=settings,
        )
        return manager.build()

    def with_overrides(self, **overrides: Any) -> BotConfig:
        """Create a new BotConfig instance with the specified overrides applied."""
        current_data = self.model_dump(exclude={"metadata", "state"})
        current_data.update(overrides)

        new_config = self.__class__(
            metadata=dict(self.metadata),
            **current_data,
        )

        new_config.state.runtime_settings = self.state.runtime_settings
        new_config.state.profile_value = self.state.profile_value
        new_config.state.overrides_snapshot = dict(self.state.overrides_snapshot)
        new_config.state.config_snapshot = self.state.config_snapshot

        return new_config


__all__ = ["BotConfig"]
