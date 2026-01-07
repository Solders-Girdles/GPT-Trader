"""
Simplified Risk Config.

Supports both spot and CFM futures risk parameters.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

RISK_CONFIG_ENV_KEYS = [
    "MAX_LEVERAGE",
    "DAILY_LOSS_LIMIT",
    "MAX_EXPOSURE_PCT",
    "MAX_POSITION_PCT_PER_SYMBOL",
    # CFM-specific keys
    "CFM_MAX_LEVERAGE",
    "CFM_MIN_LIQUIDATION_BUFFER_PCT",
    "CFM_MAX_EXPOSURE_PCT",
    "CFM_MAX_POSITION_SIZE_PCT",
]

RISK_CONFIG_ENV_ALIASES: dict[str, str] = {}


@dataclass
class RiskConfig:
    # General risk parameters
    max_leverage: int = 5
    daily_loss_limit: Decimal = Decimal("100")  # Absolute dollar limit (legacy)
    daily_loss_limit_pct: float = 0.05  # Percentage of equity (0.05 = 5%)
    max_exposure_pct: float = 0.8
    max_position_pct_per_symbol: float = 0.2

    # Added to satisfy tests/unit/gpt_trader/features/live_trade/test_config_dict_json.py
    min_liquidation_buffer_pct: float = 0.1
    leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)
    max_notional_per_symbol: dict[str, Decimal] = field(default_factory=dict)
    slippage_guard_bps: int = 50
    kill_switch_enabled: bool = False
    reduce_only_mode: bool = False

    # Day/night time window configuration
    daytime_start_utc: str = "09:00"
    daytime_end_utc: str = "17:00"
    day_leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)
    night_leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)

    # MMR (Maintenance Margin Requirement) projection configuration
    day_mmr_per_symbol: dict[str, float] = field(default_factory=dict)
    night_mmr_per_symbol: dict[str, float] = field(default_factory=dict)
    enable_pre_trade_liq_projection: bool = False

    # ==========================================================================
    # CFM (Coinbase Financial Markets) Futures Risk Parameters
    # ==========================================================================
    cfm_max_leverage: int = 5
    cfm_min_liquidation_buffer_pct: float = 0.15  # 15% buffer before liquidation
    cfm_max_exposure_pct: float = 0.8  # Max 80% of equity in CFM positions
    cfm_max_position_size_pct: float = 0.25  # Max 25% of equity per CFM position

    # CFM-specific per-symbol limits
    cfm_leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)
    cfm_max_notional_per_symbol: dict[str, Decimal] = field(default_factory=dict)

    # CFM day/night leverage caps (for overnight margin requirements)
    cfm_day_leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)
    cfm_night_leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "RiskConfig":
        """Load risk configuration from environment variables.

        Supports both legacy names (e.g., MAX_LEVERAGE) and RISK_* prefixed names
        (e.g., RISK_MAX_LEVERAGE). The RISK_* prefixed names take precedence.
        """

        def _get_env(key: str, default: str) -> str:
            """Get env var with RISK_ prefix fallback."""
            # Try RISK_* prefixed version first (canonical)
            prefixed = os.getenv(f"RISK_{key}")
            if prefixed is not None:
                return prefixed
            # Fall back to legacy non-prefixed version
            return os.getenv(key, default)

        return cls(
            max_leverage=int(_get_env("MAX_LEVERAGE", "5")),
            daily_loss_limit=Decimal(_get_env("DAILY_LOSS_LIMIT", "100")),
            daily_loss_limit_pct=float(_get_env("DAILY_LOSS_LIMIT_PCT", "0.05")),
            max_exposure_pct=float(_get_env("MAX_EXPOSURE_PCT", "0.8")),
            max_position_pct_per_symbol=float(_get_env("MAX_POSITION_PCT_PER_SYMBOL", "0.2")),
            min_liquidation_buffer_pct=float(_get_env("MIN_LIQUIDATION_BUFFER_PCT", "0.1")),
            slippage_guard_bps=int(_get_env("SLIPPAGE_GUARD_BPS", "50")),
            kill_switch_enabled=_get_env("KILL_SWITCH_ENABLED", "0") == "1",
            reduce_only_mode=_get_env("REDUCE_ONLY_MODE", "0") == "1",
            # CFM-specific parameters
            cfm_max_leverage=int(os.getenv("CFM_MAX_LEVERAGE", "5")),
            cfm_min_liquidation_buffer_pct=float(
                os.getenv("CFM_MIN_LIQUIDATION_BUFFER_PCT", "0.15")
            ),
            cfm_max_exposure_pct=float(os.getenv("CFM_MAX_EXPOSURE_PCT", "0.8")),
            cfm_max_position_size_pct=float(os.getenv("CFM_MAX_POSITION_SIZE_PCT", "0.25")),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary with Decimal handling."""

        def _convert(obj: Any) -> Any:
            if isinstance(obj, Decimal):
                return str(obj)
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            return obj

        return {k: _convert(v) for k, v in asdict(self).items()}

    @classmethod
    def from_json(cls, path: str) -> "RiskConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Type conversion for Decimal fields
        if "daily_loss_limit" in data:
            data["daily_loss_limit"] = Decimal(str(data["daily_loss_limit"]))

        if "max_notional_per_symbol" in data:
            data["max_notional_per_symbol"] = {
                k: Decimal(str(v)) for k, v in data["max_notional_per_symbol"].items()
            }

        # CFM-specific Decimal conversion
        if "cfm_max_notional_per_symbol" in data:
            data["cfm_max_notional_per_symbol"] = {
                k: Decimal(str(v)) for k, v in data["cfm_max_notional_per_symbol"].items()
            }

        return cls(**data)


__all__ = ["RiskConfig", "RISK_CONFIG_ENV_KEYS", "RISK_CONFIG_ENV_ALIASES"]
