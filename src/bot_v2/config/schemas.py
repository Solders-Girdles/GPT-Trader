"""Pydantic schemas for GPT-Trader configuration validation."""

from __future__ import annotations

from datetime import time
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from bot_v2.config.types import Profile


class BotConfigSchema(BaseModel):
    """Pydantic schema for BotConfig validation."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        validate_assignment = True

    profile: Profile
    dry_run: bool = False
    symbols: list[str] | None = None
    derivatives_enabled: bool = False
    update_interval: int = Field(default=5, ge=1)
    short_ma: int = Field(default=5, ge=1)
    long_ma: int = Field(default=20, ge=1)
    target_leverage: int = Field(default=2, ge=1)
    trailing_stop_pct: float = Field(default=0.01, ge=0)
    enable_shorts: bool = False
    max_position_size: Decimal = Field(default=Decimal("1000"), gt=0)
    max_leverage: int = Field(default=3, ge=1)
    reduce_only_mode: bool = False
    mock_broker: bool = False
    mock_fills: bool = False
    enable_order_preview: bool = False
    account_telemetry_interval: int = Field(default=300, ge=1)
    trading_window_start: time | None = None
    trading_window_end: time | None = None
    trading_days: list[str] | None = None
    daily_loss_limit: Decimal = Field(default=Decimal("0"), ge=0)
    time_in_force: Literal["GTC", "IOC", "FOK"] | None = "GTC"
    perps_enable_streaming: bool = False
    perps_stream_level: int = Field(default=1, ge=1)
    perps_paper_trading: bool = False
    perps_force_mock: bool = False
    perps_position_fraction: float | None = Field(default=None, gt=0, le=1)
    perps_skip_startup_reconcile: bool = False

    @field_validator("max_leverage", mode="before")
    @classmethod
    def validate_max_leverage(cls, v: Any) -> int:
        """Validate max_leverage is a positive integer."""
        try:
            value = int(v)
        except (TypeError, ValueError) as e:
            raise PydanticCustomError(
                "max_leverage_invalid",
                "max_leverage must be a valid integer, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if value <= 0:
            raise PydanticCustomError(
                "max_leverage_too_small",
                "max_leverage must be positive, got {value}",
                {"value": value},
            )
        return value

    @field_validator("update_interval", mode="before")
    @classmethod
    def validate_update_interval(cls, v: Any) -> int:
        """Validate update_interval is positive."""
        try:
            value = int(v)
        except (TypeError, ValueError) as e:
            raise PydanticCustomError(
                "update_interval_invalid",
                "update_interval must be a valid integer, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if value <= 0:
            raise PydanticCustomError(
                "update_interval_too_small",
                "update_interval must be positive, got {value}",
                {"value": value},
            )
        return value

    @field_validator("max_position_size", mode="before")
    @classmethod
    def validate_max_position_size(cls, v: Any) -> Decimal:
        """Validate max_position_size is positive."""
        try:
            value = Decimal(str(v))
        except (TypeError, ValueError, ArithmeticError) as e:
            raise PydanticCustomError(
                "max_position_size_invalid",
                "max_position_size must be numeric, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if value <= 0:
            raise PydanticCustomError(
                "max_position_size_too_small",
                "max_position_size must be positive, got {value}",
                {"value": str(value)},
            )
        return value

    @field_validator("daily_loss_limit", mode="before")
    @classmethod
    def validate_daily_loss_limit(cls, v: Any) -> Decimal:
        """Validate daily_loss_limit is non-negative."""
        try:
            value = Decimal(str(v))
        except (TypeError, ValueError, ArithmeticError) as e:
            raise PydanticCustomError(
                "daily_loss_limit_invalid",
                "daily_loss_limit must be numeric, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if value < 0:
            raise PydanticCustomError(
                "daily_loss_limit_negative",
                "daily_loss_limit must be non-negative, got {value}",
                {"value": str(value)},
            )
        return value

    @field_validator("symbols", mode="before")
    @classmethod
    def validate_symbols(cls, v: Any) -> list[str] | None:
        """Validate symbols is a list of non-empty strings."""
        if v is None:
            return None

        if not isinstance(v, (list, tuple)):
            raise PydanticCustomError(
                "symbols_invalid_type",
                "symbols must be a list or tuple, got {type}",
                {"type": type(v).__name__},
            )

        symbols = list(v)
        if not symbols:
            raise PydanticCustomError("symbols_empty", "symbols cannot be empty when provided")

        invalid_symbols = []
        for i, symbol in enumerate(symbols):
            if not isinstance(symbol, str) or not symbol.strip():
                invalid_symbols.append(f"[{i}]: {repr(symbol)}")

        if invalid_symbols:
            raise PydanticCustomError(
                "symbols_invalid_values",
                "symbols must contain only non-empty strings: {invalid}",
                {"invalid": ", ".join(invalid_symbols)},
            )

        return symbols

    @field_validator("time_in_force", mode="before")
    @classmethod
    def validate_time_in_force(cls, v: Any) -> str | None:
        """Validate time_in_force is supported."""
        if v is None:
            return None

        tif = str(v).upper()
        supported = {"GTC", "IOC", "FOK"}
        if tif not in supported:
            raise PydanticCustomError(
                "time_in_force_unsupported",
                "time_in_force must be one of {supported}, got {value}",
                {"supported": supported, "value": repr(v)},
            )
        return tif

    @field_validator("account_telemetry_interval", mode="before")
    @classmethod
    def validate_account_telemetry_interval(cls, v: Any) -> int:
        """Validate account_telemetry_interval is positive."""
        try:
            value = int(v)
        except (TypeError, ValueError) as e:
            raise PydanticCustomError(
                "account_telemetry_interval_invalid",
                "account_telemetry_interval must be a valid integer, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if value <= 0:
            raise PydanticCustomError(
                "account_telemetry_interval_too_small",
                "account_telemetry_interval must be positive, got {value}",
                {"value": value},
            )
        return value

    @field_validator("perps_stream_level", mode="before")
    @classmethod
    def validate_perps_stream_level(cls, v: Any) -> int:
        """Validate perps_stream_level."""
        try:
            value = int(v) if v is not None else 1
        except (TypeError, ValueError) as e:
            raise PydanticCustomError(
                "perps_stream_level_invalid",
                "perps_stream_level must be a valid integer, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if value < 1:
            raise PydanticCustomError(
                "perps_stream_level_too_small",
                "perps_stream_level must be >= 1, got {value}",
                {"value": value},
            )
        return value

    @field_validator("perps_position_fraction", mode="before")
    @classmethod
    def validate_perps_position_fraction(cls, v: Any) -> float | None:
        """Validate perps_position_fraction is in (0, 1)."""
        if v is None:
            return None

        try:
            value = float(v)
        except (TypeError, ValueError) as e:
            raise PydanticCustomError(
                "perps_position_fraction_invalid",
                "perps_position_fraction must be numeric, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if not (0 < value <= 1):
            raise PydanticCustomError(
                "perps_position_fraction_invalid_range",
                "perps_position_fraction must be in (0, 1], got {value}",
                {"value": value},
            )
        return value

    @field_validator("target_leverage", "short_ma", "long_ma", mode="before")
    @classmethod
    def validate_positive_integers(cls, v: Any, info: Any) -> int:
        """Validate these fields are positive integers."""
        field_name = info.field_name
        try:
            value = int(v)
        except (TypeError, ValueError) as e:
            raise PydanticCustomError(
                f"{field_name}_invalid",
                f"{field_name} must be a valid integer, got {{value}}: {{error}}",
                {"value": v, "error": str(e)},
            ) from e

        if value <= 0:
            raise PydanticCustomError(
                f"{field_name}_too_small",
                f"{field_name} must be positive, got {{value}}",
                {"value": value},
            )
        return value

    @field_validator("trailing_stop_pct", mode="before")
    @classmethod
    def validate_trailing_stop_pct(cls, v: Any) -> float:
        """Validate trailing_stop_pct is non-negative."""
        try:
            value = float(v)
        except (TypeError, ValueError) as e:
            raise PydanticCustomError(
                "trailing_stop_pct_invalid",
                "trailing_stop_pct must be numeric, got {value}: {error}",
                {"value": v, "error": str(e)},
            ) from e

        if value < 0:
            raise PydanticCustomError(
                "trailing_stop_pct_negative",
                "trailing_stop_pct must be non-negative, got {value}",
                {"value": value},
            )
        return value

    @model_validator(mode="after")
    def validate_profile_constraints(self) -> BotConfigSchema:
        """Validate profile-specific constraints."""
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

        # Validate MA periods make sense
        if self.short_ma >= self.long_ma:
            errors.append(
                f"[ma_periods_invalid] short_ma ({self.short_ma}) must be < long_ma ({self.long_ma})"
            )

        if errors:
            raise ValueError("; ".join(errors))

        return self


class ConfigValidationResult(BaseModel):
    """Result of configuration validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """Check if validation failed."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.warnings) > 0
