"""
Risk configuration for perpetuals live trading.

Phase 5: Risk Engine configuration only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


# --- Table-driven configuration parsers ---


def _parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    return value.lower() in ("true", "1", "yes")


def _parse_symbol_dict_int(value: str) -> dict[str, int]:
    """Parse symbol:value dictionary (BTC-PERP:10,ETH-PERP:8)."""
    return {k: int(v) for k, v in (pair.split(":") for pair in value.split(",") if ":" in pair)}


def _parse_symbol_dict_float(value: str) -> dict[str, float]:
    """Parse symbol:value dictionary with float values."""
    return {k: float(v) for k, v in (pair.split(":") for pair in value.split(",") if ":" in pair)}


def _parse_symbol_dict_decimal(value: str) -> dict[str, Decimal]:
    """Parse symbol:value dictionary with Decimal values."""
    return {
        k: Decimal(str(v)) for k, v in (pair.split(":") for pair in value.split(",") if ":" in pair)
    }


@dataclass
class EnvVarMapping:
    """Maps an environment variable to a config field with type conversion."""

    env_var: str
    field_name: str
    converter: Callable[[str], Any]
    log_parse_errors: bool = True


# Configuration mapping table - declarative approach for all env vars
ENV_VAR_MAPPINGS: list[EnvVarMapping] = [
    # Leverage controls
    EnvVarMapping("RISK_MAX_LEVERAGE", "max_leverage", int),
    EnvVarMapping(
        "RISK_LEVERAGE_MAX_PER_SYMBOL", "leverage_max_per_symbol", _parse_symbol_dict_int
    ),
    # Time-of-day schedule
    EnvVarMapping("RISK_DAYTIME_START_UTC", "daytime_start_utc", str),
    EnvVarMapping("RISK_DAYTIME_END_UTC", "daytime_end_utc", str),
    EnvVarMapping(
        "RISK_DAY_LEVERAGE_MAX_PER_SYMBOL", "day_leverage_max_per_symbol", _parse_symbol_dict_int
    ),
    EnvVarMapping(
        "RISK_NIGHT_LEVERAGE_MAX_PER_SYMBOL",
        "night_leverage_max_per_symbol",
        _parse_symbol_dict_int,
    ),
    EnvVarMapping("RISK_DAY_MMR_PER_SYMBOL", "day_mmr_per_symbol", _parse_symbol_dict_float),
    EnvVarMapping("RISK_NIGHT_MMR_PER_SYMBOL", "night_mmr_per_symbol", _parse_symbol_dict_float),
    # Liquidation safety
    EnvVarMapping("RISK_MIN_LIQUIDATION_BUFFER_PCT", "min_liquidation_buffer_pct", float),
    EnvVarMapping(
        "RISK_ENABLE_PRE_TRADE_LIQ_PROJECTION", "enable_pre_trade_liq_projection", _parse_bool
    ),
    EnvVarMapping("RISK_DEFAULT_MMR", "default_maintenance_margin_rate", float),
    # Market impact
    EnvVarMapping("RISK_ENABLE_MARKET_IMPACT_GUARD", "enable_market_impact_guard", _parse_bool),
    EnvVarMapping("RISK_MAX_MARKET_IMPACT_BPS", "max_market_impact_bps", Decimal),
    # Dynamic position sizing
    EnvVarMapping(
        "RISK_ENABLE_DYNAMIC_POSITION_SIZING", "enable_dynamic_position_sizing", _parse_bool
    ),
    EnvVarMapping("RISK_POSITION_SIZING_METHOD", "position_sizing_method", str),
    EnvVarMapping("RISK_POSITION_SIZING_MULTIPLIER", "position_sizing_multiplier", float),
    # Loss limits
    EnvVarMapping("RISK_DAILY_LOSS_LIMIT", "daily_loss_limit", Decimal),
    # Exposure controls (including legacy alias)
    EnvVarMapping("RISK_MAX_TOTAL_EXPOSURE_PCT", "max_exposure_pct", float),
    EnvVarMapping("RISK_MAX_EXPOSURE_PCT", "max_exposure_pct", float),
    EnvVarMapping("RISK_MAX_POSITION_PCT_PER_SYMBOL", "max_position_pct_per_symbol", float),
    EnvVarMapping(
        "RISK_MAX_NOTIONAL_PER_SYMBOL", "max_notional_per_symbol", _parse_symbol_dict_decimal
    ),
    # Slippage protection
    EnvVarMapping("RISK_SLIPPAGE_GUARD_BPS", "slippage_guard_bps", int),
    # Emergency controls
    EnvVarMapping("RISK_KILL_SWITCH_ENABLED", "kill_switch_enabled", _parse_bool),
    EnvVarMapping("RISK_REDUCE_ONLY_MODE", "reduce_only_mode", _parse_bool),
    # Mark price staleness
    EnvVarMapping("RISK_MAX_MARK_STALENESS_SECONDS", "max_mark_staleness_seconds", int),
    # Circuit breakers
    EnvVarMapping("RISK_ENABLE_VOLATILITY_CB", "enable_volatility_circuit_breaker", _parse_bool),
    EnvVarMapping("RISK_MAX_INTRADAY_VOL", "max_intraday_volatility_threshold", float),
    EnvVarMapping("RISK_VOL_WINDOW_PERIODS", "volatility_window_periods", int),
    EnvVarMapping("RISK_CB_COOLDOWN_MIN", "circuit_breaker_cooldown_minutes", int),
    EnvVarMapping("RISK_VOL_WARNING_THRESH", "volatility_warning_threshold", float),
    EnvVarMapping("RISK_VOL_REDUCE_ONLY_THRESH", "volatility_reduce_only_threshold", float),
    EnvVarMapping("RISK_VOL_KILL_SWITCH_THRESH", "volatility_kill_switch_threshold", float),
]


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

    # Market impact guard
    enable_market_impact_guard: bool = False
    max_market_impact_bps: Decimal = Decimal("50")

    # Dynamic position sizing
    enable_dynamic_position_sizing: bool = False
    position_sizing_method: str = "intelligent"
    position_sizing_multiplier: float = 1.0

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

    def __post_init__(self) -> None:
        # Map legacy alias to canonical exposure field, if provided
        try:
            if self.max_total_exposure_pct is not None:
                self.max_exposure_pct = float(self.max_total_exposure_pct)
        except Exception:
            pass

    @classmethod
    def from_env(cls) -> RiskConfig:
        """Load config from environment variables using table-driven approach."""
        config = cls()

        # Table-driven parsing: iterate through all mappings
        for mapping in ENV_VAR_MAPPINGS:
            val = os.getenv(mapping.env_var)
            if val is None:
                continue

            try:
                parsed_value = mapping.converter(val)
                setattr(config, mapping.field_name, parsed_value)
            except Exception as e:
                if mapping.log_parse_errors:
                    logger.warning(
                        f"Failed to parse {mapping.env_var}={val!r} for field "
                        f"{mapping.field_name}: {e.__class__.__name__}: {e}"
                    )

        return config

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
            "enable_volatility_circuit_breaker": self.enable_volatility_circuit_breaker,
            "max_intraday_volatility_threshold": self.max_intraday_volatility_threshold,
            "volatility_window_periods": self.volatility_window_periods,
            "circuit_breaker_cooldown_minutes": self.circuit_breaker_cooldown_minutes,
            "volatility_warning_threshold": self.volatility_warning_threshold,
            "volatility_reduce_only_threshold": self.volatility_reduce_only_threshold,
            "volatility_kill_switch_threshold": self.volatility_kill_switch_threshold,
        }
