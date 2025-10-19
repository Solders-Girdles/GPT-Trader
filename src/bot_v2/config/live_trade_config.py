"""
Risk configuration for perpetuals live trading.

Phase 5: Risk Engine configuration only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
else:  # pragma: no cover - runtime type alias
    RuntimeSettings = Any  # type: ignore[misc]

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.validation import (
    BooleanRule,
    DecimalRule,
    FloatRule,
    MappingRule,
    PercentageRule,
    StripStringRule,
    TimeOfDayRule,
)

from .env_utils import EnvVarError

logger = get_logger(__name__, component="risk")


def _load_runtime_settings() -> RuntimeSettings:
    from bot_v2.orchestration.runtime_settings import load_runtime_settings as _loader

    return _loader()


@dataclass
class RiskConfig:
    """Risk management configuration for perpetuals trading.

    All limits are fail-fast with explicit rejection messages.
    """

    # Leverage controls
    max_leverage: int = 5  # Global leverage cap
    leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)  # Per-symbol caps
    # Time-of-day leverage and margin (UTC-based, optional)
    daytime_start_utc: str | None = None  # e.g., "09:00"
    daytime_end_utc: str | None = None  # e.g., "17:00"
    day_leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)
    night_leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)
    day_mmr_per_symbol: dict[str, float] = field(default_factory=dict)  # maintenance margin rate
    night_mmr_per_symbol: dict[str, float] = field(default_factory=dict)

    # Liquidation safety
    min_liquidation_buffer_pct: float = 0.15  # Maintain 15% buffer from liquidation (safer default)
    enable_pre_trade_liq_projection: bool = True  # Enforce projected buffer pre-trade
    default_maintenance_margin_rate: float = 0.005  # 0.5% fallback MMR when exchange not provided

    # Loss limits
    daily_loss_limit: Decimal = Decimal("100")  # Max daily loss in USD

    # Exposure controls
    max_exposure_pct: float = 0.8  # Allow up to 80% portfolio exposure
    max_position_pct_per_symbol: float = 0.2  # Max 20% per symbol
    max_notional_per_symbol: dict[str, Decimal] = field(default_factory=dict)  # Optional hard caps

    # Slippage protection
    slippage_guard_bps: int = 50  # 50 bps = 0.5% max slippage (safer default)

    # Emergency controls
    kill_switch_enabled: bool = False  # Global halt
    reduce_only_mode: bool = False  # Only allow reducing positions

    # Mark price staleness (seconds)
    max_mark_staleness_seconds: int = 180  # Warn if >180s; halt only on severe staleness

    # Dynamic position sizing
    enable_dynamic_position_sizing: bool = False
    position_sizing_method: str = "notional"
    position_sizing_multiplier: float = 1.0

    # Market impact guard
    enable_market_impact_guard: bool = False
    max_market_impact_bps: int = 0

    # Circuit breakers (volatility)
    enable_volatility_circuit_breaker: bool = False
    max_intraday_volatility_threshold: float = 0.15  # Annualized vol threshold
    volatility_window_periods: int = 20
    circuit_breaker_cooldown_minutes: int = 30
    # Progressive levels
    volatility_warning_threshold: float = 0.15
    volatility_reduce_only_threshold: float = 0.20
    volatility_kill_switch_threshold: float = 0.25

    # --- Legacy aliases (accepted for backward compatibility) ---
    # These are accepted in constructor but not used directly. Where applicable,
    # __post_init__ maps them into the canonical fields above.
    max_total_exposure_pct: float | None = None  # alias for max_exposure_pct
    max_position_usd: Decimal | None = None  # deprecated, unused
    max_daily_loss_pct: float | None = None  # deprecated, unused

    def __post_init__(self) -> None:
        # Map legacy alias to canonical exposure field, if provided
        try:
            if self.max_total_exposure_pct is not None:
                self.max_exposure_pct = float(self.max_total_exposure_pct)
        except Exception:
            pass

    @classmethod
    def from_env(cls, *, settings: RuntimeSettings | None = None) -> RiskConfig:
        """Load config from environment variables."""
        model = _load_risk_model_from_env(settings=settings)
        return cls(**model.model_dump())  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, path: str) -> RiskConfig:
        """Load config from JSON file."""
        with open(path) as f:
            raw = json.load(f) or {}
        model = _load_risk_model_from_json(raw)
        return cls(**model.model_dump())  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence."""
        return {
            "max_leverage": self.max_leverage,
            "leverage_max_per_symbol": self.leverage_max_per_symbol,
            "daytime_start_utc": self.daytime_start_utc,
            "daytime_end_utc": self.daytime_end_utc,
            "day_leverage_max_per_symbol": self.day_leverage_max_per_symbol,
            "night_leverage_max_per_symbol": self.night_leverage_max_per_symbol,
            "day_mmr_per_symbol": self.day_mmr_per_symbol,
            "night_mmr_per_symbol": self.night_mmr_per_symbol,
            "min_liquidation_buffer_pct": self.min_liquidation_buffer_pct,
            "enable_pre_trade_liq_projection": self.enable_pre_trade_liq_projection,
            "default_maintenance_margin_rate": self.default_maintenance_margin_rate,
            "daily_loss_limit": str(self.daily_loss_limit),
            "max_exposure_pct": self.max_exposure_pct,
            "max_position_pct_per_symbol": self.max_position_pct_per_symbol,
            "max_notional_per_symbol": {k: str(v) for k, v in self.max_notional_per_symbol.items()},
            "slippage_guard_bps": self.slippage_guard_bps,
            "kill_switch_enabled": self.kill_switch_enabled,
            "reduce_only_mode": self.reduce_only_mode,
            "max_mark_staleness_seconds": self.max_mark_staleness_seconds,
            "enable_dynamic_position_sizing": self.enable_dynamic_position_sizing,
            "position_sizing_method": self.position_sizing_method,
            "position_sizing_multiplier": self.position_sizing_multiplier,
            "enable_market_impact_guard": self.enable_market_impact_guard,
            "max_market_impact_bps": self.max_market_impact_bps,
            "enable_volatility_circuit_breaker": self.enable_volatility_circuit_breaker,
            "max_intraday_volatility_threshold": self.max_intraday_volatility_threshold,
            "volatility_window_periods": self.volatility_window_periods,
            "circuit_breaker_cooldown_minutes": self.circuit_breaker_cooldown_minutes,
            "volatility_warning_threshold": self.volatility_warning_threshold,
            "volatility_reduce_only_threshold": self.volatility_reduce_only_threshold,
            "volatility_kill_switch_threshold": self.volatility_kill_switch_threshold,
        }


# ---------------------------------------------------------------------------
# Risk configuration schema metadata and validation helpers
# ---------------------------------------------------------------------------

RISK_CONFIG_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "max_leverage": ("RISK_MAX_LEVERAGE",),
    "leverage_max_per_symbol": ("RISK_LEVERAGE_MAX_PER_SYMBOL",),
    "daytime_start_utc": ("RISK_DAYTIME_START_UTC",),
    "daytime_end_utc": ("RISK_DAYTIME_END_UTC",),
    "day_leverage_max_per_symbol": ("RISK_DAY_LEVERAGE_MAX_PER_SYMBOL",),
    "night_leverage_max_per_symbol": ("RISK_NIGHT_LEVERAGE_MAX_PER_SYMBOL",),
    "day_mmr_per_symbol": ("RISK_DAY_MMR_PER_SYMBOL",),
    "night_mmr_per_symbol": ("RISK_NIGHT_MMR_PER_SYMBOL",),
    "min_liquidation_buffer_pct": ("RISK_MIN_LIQUIDATION_BUFFER_PCT",),
    "enable_pre_trade_liq_projection": ("RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION",),
    "default_maintenance_margin_rate": ("RISK_DEFAULT_MMR",),
    "daily_loss_limit": ("RISK_DAILY_LOSS_LIMIT",),
    "max_exposure_pct": ("RISK_MAX_EXPOSURE_PCT", "RISK_MAX_TOTAL_EXPOSURE_PCT"),
    "max_total_exposure_pct": ("RISK_MAX_TOTAL_EXPOSURE_PCT",),
    "max_position_usd": ("RISK_MAX_POSITION_USD",),
    "max_daily_loss_pct": ("RISK_MAX_DAILY_LOSS_PCT",),
    "max_position_pct_per_symbol": ("RISK_MAX_POSITION_PCT_PER_SYMBOL",),
    "max_notional_per_symbol": ("RISK_MAX_NOTIONAL_PER_SYMBOL",),
    "slippage_guard_bps": ("RISK_SLIPPAGE_GUARD_BPS",),
    "kill_switch_enabled": ("RISK_KILL_SWITCH_ENABLED",),
    "reduce_only_mode": ("RISK_REDUCE_ONLY_MODE",),
    "max_mark_staleness_seconds": ("RISK_MAX_MARK_STALENESS_SECONDS",),
    "enable_dynamic_position_sizing": ("RISK_ENABLE_DYNAMIC_POSITION_SIZING",),
    "position_sizing_method": ("RISK_POSITION_SIZING_METHOD",),
    "position_sizing_multiplier": ("RISK_POSITION_SIZING_MULTIPLIER",),
    "enable_market_impact_guard": ("RISK_ENABLE_MARKET_IMPACT_GUARD",),
    "max_market_impact_bps": ("RISK_MAX_MARKET_IMPACT_BPS",),
    "enable_volatility_circuit_breaker": ("RISK_ENABLE_VOLATILITY_CB",),
    "max_intraday_volatility_threshold": ("RISK_MAX_INTRADAY_VOL",),
    "volatility_window_periods": ("RISK_VOL_WINDOW_PERIODS",),
    "circuit_breaker_cooldown_minutes": ("RISK_CB_COOLDOWN_MIN",),
    "volatility_warning_threshold": ("RISK_VOL_WARNING_THRESH",),
    "volatility_reduce_only_threshold": ("RISK_VOL_REDUCE_ONLY_THRESH",),
    "volatility_kill_switch_threshold": ("RISK_VOL_KILL_SWITCH_THRESH",),
}

RISK_CONFIG_JSON_ALIASES: dict[str, tuple[str, ...]] = {
    "max_leverage": ("max_leverage",),
    "leverage_max_per_symbol": ("leverage_max_per_symbol",),
    "daytime_start_utc": ("daytime_start_utc",),
    "daytime_end_utc": ("daytime_end_utc",),
    "day_leverage_max_per_symbol": ("day_leverage_max_per_symbol",),
    "night_leverage_max_per_symbol": ("night_leverage_max_per_symbol",),
    "day_mmr_per_symbol": ("day_mmr_per_symbol",),
    "night_mmr_per_symbol": ("night_mmr_per_symbol",),
    "min_liquidation_buffer_pct": ("min_liquidation_buffer_pct",),
    "enable_pre_trade_liq_projection": ("enable_pre_trade_liq_projection",),
    "default_maintenance_margin_rate": ("default_maintenance_margin_rate",),
    "daily_loss_limit": ("daily_loss_limit",),
    "max_exposure_pct": ("max_exposure_pct",),
    "max_total_exposure_pct": ("max_total_exposure_pct",),
    "max_position_usd": ("max_position_usd",),
    "max_daily_loss_pct": ("max_daily_loss_pct",),
    "max_position_pct_per_symbol": ("max_position_pct_per_symbol",),
    "max_notional_per_symbol": ("max_notional_per_symbol",),
    "slippage_guard_bps": ("slippage_guard_bps",),
    "kill_switch_enabled": ("kill_switch_enabled",),
    "reduce_only_mode": ("reduce_only_mode",),
    "max_mark_staleness_seconds": ("max_mark_staleness_seconds",),
    "enable_dynamic_position_sizing": ("enable_dynamic_position_sizing",),
    "position_sizing_method": ("position_sizing_method",),
    "position_sizing_multiplier": ("position_sizing_multiplier",),
    "enable_market_impact_guard": ("enable_market_impact_guard",),
    "max_market_impact_bps": ("max_market_impact_bps",),
    "enable_volatility_circuit_breaker": ("enable_volatility_circuit_breaker",),
    "max_intraday_volatility_threshold": ("max_intraday_volatility_threshold",),
    "volatility_window_periods": ("volatility_window_periods",),
    "circuit_breaker_cooldown_minutes": ("circuit_breaker_cooldown_minutes",),
    "volatility_warning_threshold": ("volatility_warning_threshold",),
    "volatility_reduce_only_threshold": ("volatility_reduce_only_threshold",),
    "volatility_kill_switch_threshold": ("volatility_kill_switch_threshold",),
}

RISK_CONFIG_ENV_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(alias for aliases in RISK_CONFIG_ENV_ALIASES.values() for alias in aliases).keys()
)

RISK_CONFIG_JSON_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(
        alias for aliases in RISK_CONFIG_JSON_ALIASES.values() for alias in aliases
    ).keys()
)

_BOOL_FIELD_DEFAULTS: dict[str, bool] = {
    "enable_pre_trade_liq_projection": True,
    "kill_switch_enabled": False,
    "reduce_only_mode": False,
    "enable_dynamic_position_sizing": False,
    "enable_market_impact_guard": False,
    "enable_volatility_circuit_breaker": False,
}


def _alias_choices(field_name: str) -> AliasChoices:
    seen: list[str] = []
    for alias_group in (
        RISK_CONFIG_ENV_ALIASES.get(field_name, ()),
        RISK_CONFIG_JSON_ALIASES.get(field_name, ()),
        (field_name,),
    ):
        for alias in alias_group:
            if alias not in seen:
                seen.append(alias)
    return AliasChoices(*seen)


_BOOL_RULES = {name: BooleanRule(default=default) for name, default in _BOOL_FIELD_DEFAULTS.items()}
_INT_MAPPING_RULE = MappingRule(value_converter=int)
_FLOAT_MAPPING_RULE = MappingRule(value_converter=float)
_DECIMAL_RULE = DecimalRule()
_OPTIONAL_DECIMAL_RULE = DecimalRule(allow_none=True)
_DECIMAL_MAPPING_RULE = MappingRule(value_rule=DecimalRule())
_OPTIONAL_FLOAT_RULE = FloatRule(allow_none=True)
_FLOAT_DEFAULT_ONE_RULE = FloatRule(default=1.0, allow_none=True)
_PERCENTAGE_RULE = PercentageRule()
_OPTIONAL_PERCENTAGE_RULE = PercentageRule(allow_none=True)
_TIME_OF_DAY_RULE = TimeOfDayRule()
_POSITION_METHOD_RULE = StripStringRule(default="notional")
_DEFAULT_BOOL_RULE = BooleanRule()


class RiskConfigModel(BaseModel):
    """Validated representation matching :class:`RiskConfig` semantics."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    max_leverage: int = Field(
        default=5,
        ge=1,
        validation_alias=_alias_choices("max_leverage"),
    )
    leverage_max_per_symbol: dict[str, int] = Field(
        default_factory=dict,
        validation_alias=_alias_choices("leverage_max_per_symbol"),
    )
    daytime_start_utc: str | None = Field(
        default=None,
        validation_alias=_alias_choices("daytime_start_utc"),
    )
    daytime_end_utc: str | None = Field(
        default=None,
        validation_alias=_alias_choices("daytime_end_utc"),
    )
    day_leverage_max_per_symbol: dict[str, int] = Field(
        default_factory=dict,
        validation_alias=_alias_choices("day_leverage_max_per_symbol"),
    )
    night_leverage_max_per_symbol: dict[str, int] = Field(
        default_factory=dict,
        validation_alias=_alias_choices("night_leverage_max_per_symbol"),
    )
    day_mmr_per_symbol: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=_alias_choices("day_mmr_per_symbol"),
    )
    night_mmr_per_symbol: dict[str, float] = Field(
        default_factory=dict,
        validation_alias=_alias_choices("night_mmr_per_symbol"),
    )
    min_liquidation_buffer_pct: float = Field(
        default=0.15,
        validation_alias=_alias_choices("min_liquidation_buffer_pct"),
    )
    enable_pre_trade_liq_projection: bool = Field(
        default=True,
        validation_alias=_alias_choices("enable_pre_trade_liq_projection"),
    )
    default_maintenance_margin_rate: float = Field(
        default=0.005,
        validation_alias=_alias_choices("default_maintenance_margin_rate"),
    )
    daily_loss_limit: Decimal = Field(
        default=Decimal("100"),
        validation_alias=_alias_choices("daily_loss_limit"),
    )
    max_exposure_pct: float = Field(
        default=0.8,
        validation_alias=_alias_choices("max_exposure_pct"),
    )
    max_total_exposure_pct: float | None = Field(
        default=None,
        validation_alias=_alias_choices("max_total_exposure_pct"),
    )
    max_position_pct_per_symbol: float = Field(
        default=0.2,
        validation_alias=_alias_choices("max_position_pct_per_symbol"),
    )
    max_notional_per_symbol: dict[str, Decimal] = Field(
        default_factory=dict,
        validation_alias=_alias_choices("max_notional_per_symbol"),
    )
    max_position_usd: Decimal | None = Field(
        default=None,
        validation_alias=_alias_choices("max_position_usd"),
    )
    max_daily_loss_pct: float | None = Field(
        default=None,
        validation_alias=_alias_choices("max_daily_loss_pct"),
    )
    slippage_guard_bps: int = Field(
        default=50,
        ge=0,
        validation_alias=_alias_choices("slippage_guard_bps"),
    )
    kill_switch_enabled: bool = Field(
        default=False,
        validation_alias=_alias_choices("kill_switch_enabled"),
    )
    reduce_only_mode: bool = Field(
        default=False,
        validation_alias=_alias_choices("reduce_only_mode"),
    )
    max_mark_staleness_seconds: int = Field(
        default=180,
        ge=0,
        validation_alias=_alias_choices("max_mark_staleness_seconds"),
    )
    enable_dynamic_position_sizing: bool = Field(
        default=False,
        validation_alias=_alias_choices("enable_dynamic_position_sizing"),
    )
    position_sizing_method: str = Field(
        default="notional",
        validation_alias=_alias_choices("position_sizing_method"),
    )
    position_sizing_multiplier: float = Field(
        default=1.0,
        ge=0,
        validation_alias=_alias_choices("position_sizing_multiplier"),
    )
    enable_market_impact_guard: bool = Field(
        default=False,
        validation_alias=_alias_choices("enable_market_impact_guard"),
    )
    max_market_impact_bps: int = Field(
        default=0,
        ge=0,
        validation_alias=_alias_choices("max_market_impact_bps"),
    )
    enable_volatility_circuit_breaker: bool = Field(
        default=False,
        validation_alias=_alias_choices("enable_volatility_circuit_breaker"),
    )
    max_intraday_volatility_threshold: float = Field(
        default=0.15,
        validation_alias=_alias_choices("max_intraday_volatility_threshold"),
    )
    volatility_window_periods: int = Field(
        default=20,
        ge=1,
        validation_alias=_alias_choices("volatility_window_periods"),
    )
    circuit_breaker_cooldown_minutes: int = Field(
        default=30,
        ge=0,
        validation_alias=_alias_choices("circuit_breaker_cooldown_minutes"),
    )
    volatility_warning_threshold: float = Field(
        default=0.15,
        validation_alias=_alias_choices("volatility_warning_threshold"),
    )
    volatility_reduce_only_threshold: float = Field(
        default=0.20,
        validation_alias=_alias_choices("volatility_reduce_only_threshold"),
    )
    volatility_kill_switch_threshold: float = Field(
        default=0.25,
        validation_alias=_alias_choices("volatility_kill_switch_threshold"),
    )

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
    def _parse_bool(cls, value: object, info: ValidationInfo) -> bool:
        rule = _BOOL_RULES.get(info.field_name, _DEFAULT_BOOL_RULE)
        return rule(value, info.field_name)

    @field_validator(
        "leverage_max_per_symbol",
        "day_leverage_max_per_symbol",
        "night_leverage_max_per_symbol",
        mode="before",
    )
    @classmethod
    def _parse_int_mapping(cls, value: object, info: ValidationInfo) -> dict[str, int]:
        parsed = _INT_MAPPING_RULE(value, info.field_name)
        return {k: int(v) for k, v in parsed.items()}

    @field_validator("day_mmr_per_symbol", "night_mmr_per_symbol", mode="before")
    @classmethod
    def _parse_float_mapping(cls, value: object, info: ValidationInfo) -> dict[str, float]:
        parsed = _FLOAT_MAPPING_RULE(value, info.field_name)
        return {k: float(v) for k, v in parsed.items()}

    @field_validator("max_notional_per_symbol", mode="before")
    @classmethod
    def _parse_decimal_mapping(cls, value: object, info: ValidationInfo) -> dict[str, Decimal]:
        parsed = _DECIMAL_MAPPING_RULE(value, info.field_name)
        return {k: v if isinstance(v, Decimal) else Decimal(str(v)) for k, v in parsed.items()}

    @field_validator("max_position_usd", mode="before")
    @classmethod
    def _parse_optional_max_position(cls, value: object, info: ValidationInfo) -> Decimal | None:
        return _OPTIONAL_DECIMAL_RULE(value, info.field_name)

    @field_validator("max_daily_loss_pct", mode="before")
    @classmethod
    def _parse_optional_loss_pct(cls, value: object, info: ValidationInfo) -> float | None:
        return _OPTIONAL_FLOAT_RULE(value, info.field_name)

    @field_validator("daytime_start_utc", "daytime_end_utc", mode="before")
    @classmethod
    def _parse_time(cls, value: object, info: ValidationInfo) -> str | None:
        return _TIME_OF_DAY_RULE(value, info.field_name)

    @field_validator("daily_loss_limit", mode="before")
    @classmethod
    def _parse_decimal(cls, value: object, info: ValidationInfo) -> Decimal:
        result = _DECIMAL_RULE(value, info.field_name)
        if result is None:  # pragma: no cover - defensive guard
            raise ValueError(f"{info.field_name} must not be null")
        return result

    @field_validator("position_sizing_method", mode="before")
    @classmethod
    def _normalize_position_method(cls, value: object, info: ValidationInfo) -> str:
        return _POSITION_METHOD_RULE(value, info.field_name)

    @field_validator("position_sizing_multiplier", mode="before")
    @classmethod
    def _normalize_multiplier(cls, value: object, info: ValidationInfo) -> float:
        result = _FLOAT_DEFAULT_ONE_RULE(value, info.field_name)
        if result is None:  # pragma: no cover - defensive guard
            return 1.0
        return result

    @field_validator("max_total_exposure_pct", mode="before")
    @classmethod
    def _normalize_optional_exposure(cls, value: object, info: ValidationInfo) -> float | None:
        return cast(float | None, _OPTIONAL_FLOAT_RULE(value, info.field_name))

    @field_validator(
        "min_liquidation_buffer_pct",
        "max_exposure_pct",
        "max_position_pct_per_symbol",
        "max_intraday_volatility_threshold",
        "volatility_warning_threshold",
        "volatility_reduce_only_threshold",
        "volatility_kill_switch_threshold",
        mode="after",
    )
    @classmethod
    def _validate_percentage_fields(cls, value: float, info: ValidationInfo) -> float:
        result = _PERCENTAGE_RULE(value, info.field_name)
        assert result is not None  # Narrow type for static checkers
        return result

    @field_validator("default_maintenance_margin_rate", mode="after")
    @classmethod
    def _validate_mmr(cls, value: float, info: ValidationInfo) -> float:
        result = _PERCENTAGE_RULE(value, info.field_name)
        assert result is not None
        return result

    @field_validator("max_total_exposure_pct", mode="after")
    @classmethod
    def _validate_max_total_exposure(
        cls, value: float | None, info: ValidationInfo
    ) -> float | None:
        return cast(float | None, _OPTIONAL_PERCENTAGE_RULE(value, info.field_name))

    @model_validator(mode="after")
    def _sync_exposure_aliases(self) -> RiskConfigModel:
        if self.max_total_exposure_pct is not None:
            self.max_exposure_pct = float(self.max_total_exposure_pct)
        return self


def _load_risk_model_from_env(
    *,
    settings: RuntimeSettings | None = None,
) -> RiskConfigModel:
    runtime_settings = settings or _load_runtime_settings()
    snapshot = runtime_settings.snapshot_env(RISK_CONFIG_ENV_KEYS)
    env_values: dict[str, str] = {}
    for key, raw in snapshot.items():
        if raw is None:
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        env_values[key] = candidate
    try:
        return RiskConfigModel.model_validate(
            env_values,
            context={"source": "env", "raw_env": env_values},
        )
    except ValidationError as exc:
        env_error = _convert_env_validation_error(exc)
        logger.error(
            f"Invalid risk configuration env var {env_error.var_name}: {env_error.message}",
            operation="risk_config_load",
            status="invalid",
        )
        raise env_error from None


def _load_risk_model_from_json(raw: Mapping[str, Any]) -> RiskConfigModel:
    if not isinstance(raw, Mapping):
        raise ValueError("Risk configuration JSON must be an object")
    return RiskConfigModel.model_validate(raw, context={"source": "json"})


def _convert_env_validation_error(error: ValidationError) -> EnvVarError:
    errors = error.errors()
    if not errors:
        raise error
    first = errors[0]
    loc = first.get("loc", ("UNKNOWN_ENV",))
    var_name = str(loc[0]) if loc else "UNKNOWN_ENV"
    message = first.get("msg", "invalid value")
    raw_value = first.get("input")
    captured = raw_value if isinstance(raw_value, str) else None
    return EnvVarError(var_name, message, captured)
