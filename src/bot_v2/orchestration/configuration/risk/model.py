"""Pydantic risk configuration model."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # pragma: no cover - import for static analysis only
    from bot_v2.orchestration.runtime_settings import RuntimeSettings

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FieldValidationInfo,
    field_validator,
    model_validator,
)

from .constants import RISK_CONFIG_ENV_ALIASES
from .loaders import load_from_env, load_from_json, load_from_legacy
from .rules import (
    BOOL_RULE,
    DECIMAL_RULE,
    FLOAT_RULE,
    INT_RULE,
    MAPPING_RULE,
    PCT_RULE,
    STRING_RULE,
    TIME_RULE,
    apply_rule,
)


class RiskConfig(BaseModel):
    """Risk management configuration for perpetuals trading."""

    max_leverage: int = Field(default=5, description="Global leverage cap")
    leverage_max_per_symbol: dict[str, int] = Field(
        default_factory=dict, description="Per-symbol leverage caps"
    )

    daytime_start_utc: str | None = Field(
        default=None, description="Daytime start UTC, e.g., '09:00'"
    )
    daytime_end_utc: str | None = Field(default=None, description="Daytime end UTC, e.g., '17:00'")
    day_leverage_max_per_symbol: dict[str, int] = Field(
        default_factory=dict, description="Daytime leverage caps per symbol"
    )
    night_leverage_max_per_symbol: dict[str, int] = Field(
        default_factory=dict, description="Nighttime leverage caps per symbol"
    )
    day_mmr_per_symbol: dict[str, float] = Field(
        default_factory=dict, description="Daytime maintenance margin rate per symbol"
    )
    night_mmr_per_symbol: dict[str, float] = Field(
        default_factory=dict, description="Nighttime maintenance margin rate per symbol"
    )

    min_liquidation_buffer_pct: float = Field(
        default=0.15,
        description="Maintain 15% buffer from liquidation (safer default)",
    )
    enable_pre_trade_liq_projection: bool = Field(
        default=True,
        description="Enforce projected buffer pre-trade",
    )
    default_maintenance_margin_rate: float = Field(
        default=0.005,
        description="0.5% fallback MMR when exchange not provided",
    )

    daily_loss_limit: Decimal = Field(
        default=Decimal("100"),
        description="Max daily loss in USD",
    )

    max_exposure_pct: float = Field(
        default=0.8,
        description="Allow up to 80% portfolio exposure",
    )
    max_position_pct_per_symbol: float = Field(
        default=0.2,
        description="Max 20% per symbol",
    )
    max_notional_per_symbol: dict[str, Decimal] = Field(
        default_factory=dict,
        description="Optional hard notional caps per symbol",
    )

    slippage_guard_bps: int = Field(
        default=50,
        description="50 bps = 0.5% max slippage (safer default)",
    )

    kill_switch_enabled: bool = Field(
        default=False,
        description="Global halt",
    )
    reduce_only_mode: bool = Field(
        default=False,
        description="Only allow reducing positions",
    )

    max_mark_staleness_seconds: int = Field(
        default=180,
        description="Warn if >180s; halt only on severe staleness",
    )

    enable_dynamic_position_sizing: bool = Field(
        default=False,
        description="Enable dynamic position sizing based on volatility",
    )
    position_sizing_method: str = Field(
        default="notional",
        description="Position sizing method: 'notional' or 'volatility'",
    )
    position_sizing_multiplier: float = Field(
        default=1.0,
        description="Multiplier for position sizing calculations",
    )

    enable_market_impact_guard: bool = Field(
        default=False,
        description="Enable market impact protection",
    )
    max_market_impact_bps: int = Field(
        default=0,
        description="Maximum allowed market impact in basis points",
    )

    enable_volatility_circuit_breaker: bool = Field(
        default=False,
        description="Enable volatility-based circuit breakers",
    )
    max_intraday_volatility_threshold: float = Field(
        default=0.15,
        description="Annualized vol threshold for circuit breaker",
    )
    volatility_window_periods: int = Field(
        default=20,
        description="Volatility calculation window in periods",
    )
    circuit_breaker_cooldown_minutes: int = Field(
        default=30,
        description="Cooldown period after circuit breaker triggers",
    )
    volatility_warning_threshold: float = Field(
        default=0.15,
        description="Volatility threshold for warnings",
    )
    volatility_reduce_only_threshold: float = Field(
        default=0.20,
        description="Volatility threshold for reduce-only mode",
    )
    volatility_kill_switch_threshold: float = Field(
        default=0.25,
        description="Volatility threshold for kill switch",
    )

    max_total_exposure_pct: float | None = Field(
        default=None,
        description="Legacy alias for max_exposure_pct (deprecated)",
    )
    max_position_usd: Decimal | None = Field(
        default=None,
        description="Legacy field (deprecated, unused)",
    )
    max_daily_loss_pct: float | None = Field(
        default=None,
        description="Legacy field (deprecated, unused)",
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    _ENV_ALIASES: ClassVar[dict[str, tuple[str, ...]]] = RISK_CONFIG_ENV_ALIASES

    @property
    def max_position_size(self) -> Decimal | None:
        return getattr(self, "max_position_usd", None)

    @max_position_size.setter
    def max_position_size(self, value: Any) -> None:
        if value is None:
            self.max_position_usd = None
            return
        try:
            self.max_position_usd = Decimal(str(value))
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ValueError("max_position_size must be numeric") from exc

    @field_validator("max_leverage", mode="before")
    @classmethod
    def _validate_max_leverage(cls, value: Any) -> int:
        result = apply_rule(
            INT_RULE,
            value,
            field_label="max_leverage",
            error_code="max_leverage_invalid",
            error_template="max_leverage must be a valid integer, got {value}: {error}",
        )
        if result <= 0:
            raise ValueError("max_leverage must be positive")
        return int(result)

    @field_validator(
        "leverage_max_per_symbol",
        "day_leverage_max_per_symbol",
        "night_leverage_max_per_symbol",
        mode="before",
    )
    @classmethod
    def _validate_leverage_symbol_caps(cls, value: Any) -> dict[str, int]:
        if value is None:
            return {}
        result = apply_rule(
            MAPPING_RULE,
            value,
            field_label="leverage caps",
            error_code="leverage_caps_invalid",
            error_template="leverage caps must be a mapping, got {value}: {error}",
        )
        validated_caps = {}
        for symbol, leverage in result.items():
            leverage_int = int(
                apply_rule(
                    INT_RULE,
                    leverage,
                    field_label=f"leverage_{symbol}",
                    error_code="symbol_leverage_invalid",
                    error_template="leverage for {symbol} must be integer, got {value}: {error}",
                )
            )
            if leverage_int > 0:
                validated_caps[symbol] = leverage_int
        return validated_caps

    @field_validator("daytime_start_utc", "daytime_end_utc", mode="before")
    @classmethod
    def _validate_time_of_day(cls, value: Any) -> str | None:
        if value is None:
            return None
        return apply_rule(
            TIME_RULE,
            value,
            field_label="time_of_day",
            error_code="time_format_invalid",
            error_template="Time must be HH:MM format, got {value}: {error}",
        )

    @field_validator(
        "min_liquidation_buffer_pct",
        "max_exposure_pct",
        "max_total_exposure_pct",
        "max_position_pct_per_symbol",
        mode="before",
    )
    @classmethod
    def _validate_percentages(cls, value: Any, info: FieldValidationInfo) -> float | None:
        if value is None:
            return None
        result = apply_rule(
            PCT_RULE,
            value,
            field_label=info.field_name,
            error_code=f"{info.field_name}_invalid",
            error_template=f"{info.field_name} must be a percentage, got {{value}}: {{error}}",
        )
        if not 0 <= result <= 1:
            raise ValueError(f"{info.field_name} must be between 0 and 1")
        return float(result)

    @field_validator("daily_loss_limit", "max_notional_per_symbol", mode="before")
    @classmethod
    def _validate_decimal_amounts(cls, value: Any, info: FieldValidationInfo) -> Any:
        if info.field_name == "daily_loss_limit":
            result = apply_rule(
                DECIMAL_RULE,
                value,
                field_label="daily_loss_limit",
                error_code="daily_loss_limit_invalid",
                error_template="daily_loss_limit must be numeric, got {value}: {error}",
            )
            if result < 0:
                raise ValueError("daily_loss_limit must be non-negative")
            return result

        if value is None:
            return {}
        result = apply_rule(
            MAPPING_RULE,
            value,
            field_label="max_notional_per_symbol",
            error_code="max_notional_invalid",
            error_template="max_notional_per_symbol must be a mapping, got {value}: {error}",
        )
        validated_caps = {}
        for symbol, notional in result.items():
            notional_decimal = Decimal(
                str(
                    apply_rule(
                        DECIMAL_RULE,
                        notional,
                        field_label=f"max_notional_{symbol}",
                        error_code="symbol_notional_invalid",
                        error_template="max notional for {symbol} must be numeric, got {value}: {error}",
                    )
                )
            )
            if notional_decimal > 0:
                validated_caps[symbol] = notional_decimal
        return validated_caps

    @field_validator(
        "slippage_guard_bps",
        "max_market_impact_bps",
        "max_mark_staleness_seconds",
        mode="before",
    )
    @classmethod
    def _validate_positive_integers(cls, value: Any, info: FieldValidationInfo) -> int:
        result = apply_rule(
            INT_RULE,
            value,
            field_label=info.field_name,
            error_code=f"{info.field_name}_invalid",
            error_template=f"{info.field_name} must be a valid integer, got {{value}}: {{error}}",
        )
        if result < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return int(result)

    @field_validator(
        "volatility_window_periods",
        "circuit_breaker_cooldown_minutes",
        mode="before",
    )
    @classmethod
    def _validate_positive_time_periods(cls, value: Any, info: FieldValidationInfo) -> int:
        result = apply_rule(
            INT_RULE,
            value,
            field_label=info.field_name,
            error_code=f"{info.field_name}_invalid",
            error_template=f"{info.field_name} must be a positive integer, got {{value}}: {{error}}",
        )
        if result <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return int(result)

    @field_validator("position_sizing_method", mode="before")
    @classmethod
    def _validate_position_sizing_method(cls, value: Any) -> str:
        result = apply_rule(
            STRING_RULE,
            value,
            field_label="position_sizing_method",
            error_code="position_sizing_method_invalid",
            error_template="position_sizing_method must be a string, got {value}: {error}",
        )
        return result.strip()

    @field_validator(
        "volatility_warning_threshold",
        "volatility_reduce_only_threshold",
        "volatility_kill_switch_threshold",
        mode="before",
    )
    @classmethod
    def _validate_volatility_thresholds(cls, value: Any, info: FieldValidationInfo) -> float:
        result = apply_rule(
            FLOAT_RULE,
            value,
            field_label=info.field_name,
            error_code=f"{info.field_name}_invalid",
            error_template=f"{info.field_name} must be numeric, got {{value}}: {{error}}",
        )
        if result < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return float(result)

    @field_validator("position_sizing_multiplier", mode="before")
    @classmethod
    def _validate_position_sizing_multiplier(cls, value: Any) -> float:
        result = apply_rule(
            FLOAT_RULE,
            value,
            field_label="position_sizing_multiplier",
            error_code="position_sizing_multiplier_invalid",
            error_template="position_sizing_multiplier must be numeric, got {value}: {error}",
        )
        if result <= 0:
            raise ValueError("position_sizing_multiplier must be positive")
        return float(result)

    @field_validator(
        "enable_pre_trade_liq_projection",
        "kill_switch_enabled",
        "reduce_only_mode",
        "enable_dynamic_position_sizing",
        "enable_market_impact_guard",
        "enable_volatility_circuit_breaker",
        mode="before",
    )
    @classmethod
    def _validate_booleans(cls, value: Any) -> bool:
        result = apply_rule(
            BOOL_RULE,
            value,
            field_label="boolean_field",
            error_code="boolean_invalid",
            error_template="must be boolean, got {value}: {error}",
        )
        return bool(result)

    @model_validator(mode="after")
    def _apply_legacy_aliases(self) -> "RiskConfig":
        if self.max_total_exposure_pct is not None:
            object.__setattr__(self, "max_exposure_pct", float(self.max_total_exposure_pct))
        return self

    def to_dict(self) -> dict[str, Any]:
        def _convert(value: Any) -> Any:
            if isinstance(value, Decimal):
                text = format(value, "f")
                if "." in text:
                    text = text.rstrip("0").rstrip(".")
                return text
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            return value

        data = self.model_dump(exclude_none=True)
        return {key: _convert(val) for key, val in data.items()}

    def to_legacy_format(self) -> Any:
        from bot_v2.config.live_trade_config import RiskConfig as LegacyRiskConfig

        return LegacyRiskConfig(**self.model_dump())

    def copy_with_overrides(self, **overrides: Any) -> RiskConfig:
        current_data = self.model_dump()
        current_data.update(overrides)
        return self.__class__(**current_data)

    @classmethod
    def from_env(cls, *, settings: RuntimeSettings | None = None) -> RiskConfig:
        return load_from_env(cls, settings=settings)

    @classmethod
    def from_json(cls, path: str | Path) -> RiskConfig:
        return load_from_json(cls, path)

    @classmethod
    def from_legacy_config(cls, legacy_config: Any) -> RiskConfig:
        return load_from_legacy(cls, legacy_config)


__all__ = ["RiskConfig"]
