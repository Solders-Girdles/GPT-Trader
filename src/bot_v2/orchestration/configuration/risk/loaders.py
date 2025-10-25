"""Risk configuration loading helpers."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from bot_v2.config.env_utils import (
    EnvVarError,
    get_env_bool,
    get_env_decimal,
    get_env_float,
    get_env_int,
    parse_env_mapping,
)
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings

from .constants import RISK_CONFIG_ENV_ALIASES

RiskConfigType = TypeVar("RiskConfigType", bound="RiskConfigProtocol")


class RiskConfigProtocol:
    """Protocol describing the constructor signature needed for loaders."""

    def __class_getitem__(cls, item: Any):  # pragma: no cover - typing helper
        return cls

    def __init__(self, **data: Any) -> None:  # pragma: no cover - typing helper
        ...


def load_from_json(model_cls: type[RiskConfigType], path: str | Path) -> RiskConfigType:
    """Load risk configuration from JSON file."""
    with open(path) as f:
        raw = json.load(f) or {}
    return model_cls(**raw)


def load_from_legacy(model_cls: type[RiskConfigType], legacy_config: Any) -> RiskConfigType:
    """Create modern RiskConfig from legacy dataclass instance."""
    if hasattr(legacy_config, "__dict__"):
        return model_cls(**legacy_config.__dict__)
    raise ValueError("legacy_config must be a dataclass or similar object")


def load_from_env(
    model_cls: type[RiskConfigType],
    *,
    settings: RuntimeSettings | None = None,
) -> RiskConfigType:
    """Load risk configuration from environment variables."""
    settings = settings or load_runtime_settings()
    raw_env = getattr(settings, "raw_env", {})

    def _has(var: str) -> bool:
        return var in raw_env and raw_env[var] not in (None, "")

    risk_data: dict[str, Any] = {}

    def _set_int(var: str, field: str) -> None:
        if _has(var):
            risk_data[field] = get_env_int(var, settings=settings)

    def _set_decimal(var: str, field: str) -> None:
        if _has(var):
            risk_data[field] = get_env_decimal(var, settings=settings)

    def _set_float(var: str, field: str) -> None:
        if _has(var):
            risk_data[field] = get_env_float(var, settings=settings)

    def _set_percentage(var: str, field: str) -> None:
        if not _has(var):
            return
        value = get_env_float(var, settings=settings)
        if value is not None:
            if not 0 <= value <= 1:
                raise EnvVarError(var, "must be between 0 and 1", str(raw_env.get(var)))
            risk_data[field] = value

    def _set_bool(var: str, field: str) -> None:
        if _has(var):
            risk_data[field] = get_env_bool(var, settings=settings)

    def _set_mapping(var: str, field: str, caster: Any) -> None:
        if _has(var):
            mapping = parse_env_mapping(var, caster, settings=settings)
            risk_data[field] = mapping

    def _set_string(var: str, field: str) -> None:
        if not _has(var):
            return
        value = raw_env.get(var)
        if isinstance(value, str):
            value = value.strip()
        risk_data[field] = value

    _set_int("RISK_MAX_LEVERAGE", "max_leverage")
    _set_mapping("RISK_LEVERAGE_MAX_PER_SYMBOL", "leverage_max_per_symbol", int)
    _set_string("RISK_DAYTIME_START_UTC", "daytime_start_utc")
    _set_string("RISK_DAYTIME_END_UTC", "daytime_end_utc")
    _set_mapping("RISK_DAY_LEVERAGE_MAX_PER_SYMBOL", "day_leverage_max_per_symbol", int)
    _set_mapping("RISK_NIGHT_LEVERAGE_MAX_PER_SYMBOL", "night_leverage_max_per_symbol", int)
    _set_mapping("RISK_DAY_MMR_PER_SYMBOL", "day_mmr_per_symbol", float)
    _set_mapping("RISK_NIGHT_MMR_PER_SYMBOL", "night_mmr_per_symbol", float)
    _set_percentage("RISK_MIN_LIQUIDATION_BUFFER_PCT", "min_liquidation_buffer_pct")
    _set_bool("RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION", "enable_pre_trade_liq_projection")
    _set_float("RISK_DEFAULT_MMR", "default_maintenance_margin_rate")
    _set_decimal("RISK_DAILY_LOSS_LIMIT", "daily_loss_limit")
    _set_percentage("RISK_MAX_EXPOSURE_PCT", "max_exposure_pct")
    if _has("RISK_MAX_TOTAL_EXPOSURE_PCT"):
        value = get_env_float("RISK_MAX_TOTAL_EXPOSURE_PCT", settings=settings)
        if value is not None:
            if not 0 <= value <= 1:
                raise EnvVarError(
                    "RISK_MAX_TOTAL_EXPOSURE_PCT",
                    "must be between 0 and 1",
                    str(raw_env.get("RISK_MAX_TOTAL_EXPOSURE_PCT")),
                )
            risk_data["max_total_exposure_pct"] = value
            risk_data.setdefault("max_exposure_pct", value)
    _set_decimal("RISK_MAX_POSITION_USD", "max_position_usd")
    _set_percentage("RISK_MAX_DAILY_LOSS_PCT", "max_daily_loss_pct")
    _set_percentage("RISK_MAX_POSITION_PCT_PER_SYMBOL", "max_position_pct_per_symbol")
    if _has("RISK_MAX_NOTIONAL_PER_SYMBOL"):
        mapping = parse_env_mapping("RISK_MAX_NOTIONAL_PER_SYMBOL", Decimal, settings=settings)
        risk_data["max_notional_per_symbol"] = mapping
    _set_int("RISK_SLIPPAGE_GUARD_BPS", "slippage_guard_bps")
    _set_bool("RISK_KILL_SWITCH_ENABLED", "kill_switch_enabled")
    _set_bool("RISK_REDUCE_ONLY_MODE", "reduce_only_mode")
    _set_int("RISK_MAX_MARK_STALENESS_SECONDS", "max_mark_staleness_seconds")
    _set_bool("RISK_ENABLE_DYNAMIC_POSITION_SIZING", "enable_dynamic_position_sizing")
    if _has("RISK_POSITION_SIZING_METHOD"):
        risk_data["position_sizing_method"] = raw_env.get("RISK_POSITION_SIZING_METHOD")
    _set_float("RISK_POSITION_SIZING_MULTIPLIER", "position_sizing_multiplier")
    _set_bool("RISK_ENABLE_MARKET_IMPACT_GUARD", "enable_market_impact_guard")
    _set_int("RISK_MAX_MARKET_IMPACT_BPS", "max_market_impact_bps")
    _set_bool("RISK_ENABLE_VOLATILITY_CB", "enable_volatility_circuit_breaker")
    _set_float("RISK_MAX_INTRADAY_VOL", "max_intraday_volatility_threshold")
    _set_int("RISK_VOL_WINDOW_PERIODS", "volatility_window_periods")
    _set_int("RISK_CB_COOLDOWN_MIN", "circuit_breaker_cooldown_minutes")
    _set_float("RISK_VOL_WARNING_THRESH", "volatility_warning_threshold")
    _set_float("RISK_VOL_REDUCE_ONLY_THRESH", "volatility_reduce_only_threshold")
    _set_float("RISK_VOL_KILL_SWITCH_THRESH", "volatility_kill_switch_threshold")

    try:
        return model_cls(**risk_data)
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first = errors[0]
            loc = first.get("loc", ("unknown",))
            field_name = loc[0] if loc else "unknown"
            aliases = RISK_CONFIG_ENV_ALIASES.get(field_name, (field_name,))
            env_var = next((alias for alias in aliases if alias in raw_env), aliases[0])
            message = first.get("msg", "invalid value")
            raise EnvVarError(env_var, message, str(raw_env.get(env_var))) from None
        raise


__all__ = ["load_from_env", "load_from_json", "load_from_legacy"]
