"""
Risk configuration for perpetuals live trading.

Phase 5: Risk Engine configuration only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bot_v2.orchestration.runtime_settings import RuntimeSettings
else:  # pragma: no cover - runtime type alias
    RuntimeSettings = Any  # type: ignore[misc]

from .env_utils import (
    get_env_bool,
    get_env_decimal,
    get_env_float,
    get_env_int,
    parse_env_mapping,
)


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
        runtime_settings = settings or _load_runtime_settings()
        raw_env = runtime_settings.raw_env

        # Leverage controls
        leverage_max_per_symbol = parse_env_mapping(
            "RISK_LEVERAGE_MAX_PER_SYMBOL",
            int,
            settings=runtime_settings,
        )
        day_leverage_max_per_symbol = parse_env_mapping(
            "RISK_DAY_LEVERAGE_MAX_PER_SYMBOL",
            int,
            settings=runtime_settings,
        )
        night_leverage_max_per_symbol = parse_env_mapping(
            "RISK_NIGHT_LEVERAGE_MAX_PER_SYMBOL",
            int,
            settings=runtime_settings,
        )

        # Margin rates
        day_mmr_per_symbol = parse_env_mapping(
            "RISK_DAY_MMR_PER_SYMBOL",
            float,
            settings=runtime_settings,
        )
        night_mmr_per_symbol = parse_env_mapping(
            "RISK_NIGHT_MMR_PER_SYMBOL",
            float,
            settings=runtime_settings,
        )

        # Position limits
        max_notional_per_symbol = parse_env_mapping(
            "RISK_MAX_NOTIONAL_PER_SYMBOL",
            lambda raw: Decimal(raw),
            settings=runtime_settings,
        )

        pre_trade_liq_projection = get_env_bool(
            "RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION",
            default=True,
            settings=runtime_settings,
        )
        if pre_trade_liq_projection is None:
            pre_trade_liq_projection = True

        return cls(
            # Leverage controls
            max_leverage=get_env_int("RISK_MAX_LEVERAGE", settings=runtime_settings) or 5,
            leverage_max_per_symbol=leverage_max_per_symbol,
            # Time-of-day schedule
            daytime_start_utc=raw_env.get("RISK_DAYTIME_START_UTC"),
            daytime_end_utc=raw_env.get("RISK_DAYTIME_END_UTC"),
            day_leverage_max_per_symbol=day_leverage_max_per_symbol,
            night_leverage_max_per_symbol=night_leverage_max_per_symbol,
            day_mmr_per_symbol=day_mmr_per_symbol,
            night_mmr_per_symbol=night_mmr_per_symbol,
            # Liquidation safety
            min_liquidation_buffer_pct=(
                get_env_float("RISK_MIN_LIQUIDATION_BUFFER_PCT", settings=runtime_settings) or 0.15
            ),
            enable_pre_trade_liq_projection=pre_trade_liq_projection,
            default_maintenance_margin_rate=(
                get_env_float("RISK_DEFAULT_MMR", settings=runtime_settings) or 0.005
            ),
            # Loss limits
            daily_loss_limit=(
                get_env_decimal("RISK_DAILY_LOSS_LIMIT", settings=runtime_settings)
                or Decimal("100")
            ),
            # Exposure controls (with legacy support)
            max_exposure_pct=(
                get_env_float("RISK_MAX_TOTAL_EXPOSURE_PCT", settings=runtime_settings)
                or get_env_float("RISK_MAX_EXPOSURE_PCT", settings=runtime_settings)
                or 0.8
            ),
            max_position_pct_per_symbol=(
                get_env_float("RISK_MAX_POSITION_PCT_PER_SYMBOL", settings=runtime_settings) or 0.2
            ),
            max_notional_per_symbol=max_notional_per_symbol,
            # Slippage protection
            slippage_guard_bps=(
                get_env_int("RISK_SLIPPAGE_GUARD_BPS", settings=runtime_settings) or 50
            ),
            # Emergency controls
            kill_switch_enabled=(
                get_env_bool("RISK_KILL_SWITCH_ENABLED", settings=runtime_settings) or False
            ),
            reduce_only_mode=(
                get_env_bool("RISK_REDUCE_ONLY_MODE", settings=runtime_settings) or False
            ),
            # Mark price staleness
            max_mark_staleness_seconds=(
                get_env_int("RISK_MAX_MARK_STALENESS_SECONDS", settings=runtime_settings) or 180
            ),
            # Dynamic position sizing
            enable_dynamic_position_sizing=(
                get_env_bool("RISK_ENABLE_DYNAMIC_POSITION_SIZING", settings=runtime_settings)
                or False
            ),
            position_sizing_method=raw_env.get("RISK_POSITION_SIZING_METHOD", "notional"),
            position_sizing_multiplier=(
                get_env_float("RISK_POSITION_SIZING_MULTIPLIER", settings=runtime_settings) or 1.0
            ),
            # Market impact guard
            enable_market_impact_guard=(
                get_env_bool("RISK_ENABLE_MARKET_IMPACT_GUARD", settings=runtime_settings) or False
            ),
            max_market_impact_bps=(
                get_env_int("RISK_MAX_MARKET_IMPACT_BPS", settings=runtime_settings) or 0
            ),
            # Circuit breakers
            enable_volatility_circuit_breaker=(
                get_env_bool("RISK_ENABLE_VOLATILITY_CB", settings=runtime_settings) or False
            ),
            max_intraday_volatility_threshold=(
                get_env_float("RISK_MAX_INTRADAY_VOL", settings=runtime_settings) or 0.15
            ),
            volatility_window_periods=(
                get_env_int("RISK_VOL_WINDOW_PERIODS", settings=runtime_settings) or 20
            ),
            circuit_breaker_cooldown_minutes=(
                get_env_int("RISK_CB_COOLDOWN_MIN", settings=runtime_settings) or 30
            ),
            volatility_warning_threshold=(
                get_env_float("RISK_VOL_WARNING_THRESH", settings=runtime_settings) or 0.15
            ),
            volatility_reduce_only_threshold=(
                get_env_float("RISK_VOL_REDUCE_ONLY_THRESH", settings=runtime_settings) or 0.20
            ),
            volatility_kill_switch_threshold=(
                get_env_float("RISK_VOL_KILL_SWITCH_THRESH", settings=runtime_settings) or 0.25
            ),
        )

    @classmethod
    def from_json(cls, path: str) -> RiskConfig:
        """Load config from JSON file."""
        with open(path) as f:
            raw = json.load(f) or {}

        # Normalize types to match constructor expectations
        data: dict[str, object] = {}

        # Simple passthroughs with safe casts
        if "max_leverage" in raw:
            data["max_leverage"] = int(raw["max_leverage"])
        if "min_liquidation_buffer_pct" in raw:
            data["min_liquidation_buffer_pct"] = float(raw["min_liquidation_buffer_pct"])
        if "enable_pre_trade_liq_projection" in raw:
            data["enable_pre_trade_liq_projection"] = bool(raw["enable_pre_trade_liq_projection"])
        if "default_maintenance_margin_rate" in raw:
            data["default_maintenance_margin_rate"] = float(raw["default_maintenance_margin_rate"])
        if "daily_loss_limit" in raw:
            data["daily_loss_limit"] = Decimal(str(raw["daily_loss_limit"]))
        if "max_exposure_pct" in raw:
            data["max_exposure_pct"] = float(raw["max_exposure_pct"])
        if "max_position_pct_per_symbol" in raw:
            data["max_position_pct_per_symbol"] = float(raw["max_position_pct_per_symbol"])
        # Legacy alias
        if "max_total_exposure_pct" in raw:
            data["max_exposure_pct"] = float(raw["max_total_exposure_pct"])
        if "slippage_guard_bps" in raw:
            data["slippage_guard_bps"] = int(raw["slippage_guard_bps"])
        if "kill_switch_enabled" in raw:
            data["kill_switch_enabled"] = bool(raw["kill_switch_enabled"])
        if "reduce_only_mode" in raw:
            data["reduce_only_mode"] = bool(raw["reduce_only_mode"])
        if "max_mark_staleness_seconds" in raw:
            data["max_mark_staleness_seconds"] = int(raw["max_mark_staleness_seconds"])
        if "enable_dynamic_position_sizing" in raw:
            data["enable_dynamic_position_sizing"] = bool(raw["enable_dynamic_position_sizing"])
        if "position_sizing_method" in raw:
            data["position_sizing_method"] = str(raw["position_sizing_method"])
        if "position_sizing_multiplier" in raw:
            data["position_sizing_multiplier"] = float(raw["position_sizing_multiplier"])
        if "enable_market_impact_guard" in raw:
            data["enable_market_impact_guard"] = bool(raw["enable_market_impact_guard"])
        if "max_market_impact_bps" in raw:
            data["max_market_impact_bps"] = int(raw["max_market_impact_bps"])
        if "enable_volatility_circuit_breaker" in raw:
            data["enable_volatility_circuit_breaker"] = bool(
                raw["enable_volatility_circuit_breaker"]
            )
        if "max_intraday_volatility_threshold" in raw:
            data["max_intraday_volatility_threshold"] = float(
                raw["max_intraday_volatility_threshold"]
            )
        if "volatility_window_periods" in raw:
            data["volatility_window_periods"] = int(raw["volatility_window_periods"])
        if "circuit_breaker_cooldown_minutes" in raw:
            data["circuit_breaker_cooldown_minutes"] = int(raw["circuit_breaker_cooldown_minutes"])
        if "volatility_warning_threshold" in raw:
            data["volatility_warning_threshold"] = float(raw["volatility_warning_threshold"])
        if "volatility_reduce_only_threshold" in raw:
            data["volatility_reduce_only_threshold"] = float(
                raw["volatility_reduce_only_threshold"]
            )
        if "volatility_kill_switch_threshold" in raw:
            data["volatility_kill_switch_threshold"] = float(
                raw["volatility_kill_switch_threshold"]
            )

        # Mappings
        if "leverage_max_per_symbol" in raw and isinstance(raw["leverage_max_per_symbol"], dict):
            data["leverage_max_per_symbol"] = {
                str(k): int(v) for k, v in raw["leverage_max_per_symbol"].items()
            }
        if "max_notional_per_symbol" in raw and isinstance(raw["max_notional_per_symbol"], dict):
            data["max_notional_per_symbol"] = {
                str(k): Decimal(str(v)) for k, v in raw["max_notional_per_symbol"].items()
            }
        # Optional day/night schedule
        if "daytime_start_utc" in raw:
            data["daytime_start_utc"] = (
                str(raw["daytime_start_utc"]) if raw["daytime_start_utc"] else None
            )
        if "daytime_end_utc" in raw:
            data["daytime_end_utc"] = (
                str(raw["daytime_end_utc"]) if raw["daytime_end_utc"] else None
            )
        if "day_leverage_max_per_symbol" in raw and isinstance(
            raw["day_leverage_max_per_symbol"], dict
        ):
            data["day_leverage_max_per_symbol"] = {
                str(k): int(v) for k, v in raw["day_leverage_max_per_symbol"].items()
            }
        if "night_leverage_max_per_symbol" in raw and isinstance(
            raw["night_leverage_max_per_symbol"], dict
        ):
            data["night_leverage_max_per_symbol"] = {
                str(k): int(v) for k, v in raw["night_leverage_max_per_symbol"].items()
            }
        if "day_mmr_per_symbol" in raw and isinstance(raw["day_mmr_per_symbol"], dict):
            data["day_mmr_per_symbol"] = {
                str(k): float(v) for k, v in raw["day_mmr_per_symbol"].items()
            }
        if "night_mmr_per_symbol" in raw and isinstance(raw["night_mmr_per_symbol"], dict):
            data["night_mmr_per_symbol"] = {
                str(k): float(v) for k, v in raw["night_mmr_per_symbol"].items()
            }

        # Fill any missing keys from raw directly (best-effort)
        for k, v in raw.items():
            if k not in data:
                data[k] = v

        return cls(**data)  # type: ignore[arg-type]

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
