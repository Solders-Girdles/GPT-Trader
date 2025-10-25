"""Constants used by risk configuration."""

from __future__ import annotations

from pathlib import Path

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

RISK_CONFIG_ENV_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(alias for aliases in RISK_CONFIG_ENV_ALIASES.values() for alias in aliases).keys()
)

DEFAULT_RISK_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "config" / "risk" / "perpetuals.json"
)

__all__ = ["RISK_CONFIG_ENV_ALIASES", "RISK_CONFIG_ENV_KEYS", "DEFAULT_RISK_CONFIG_PATH"]
