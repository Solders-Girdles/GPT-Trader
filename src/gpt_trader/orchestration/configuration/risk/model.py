"""
Simplified Risk Config.
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
]

RISK_CONFIG_ENV_ALIASES = {}


@dataclass
class RiskConfig:
    max_leverage: int = 5
    daily_loss_limit: Decimal = Decimal("100")
    max_exposure_pct: float = 0.8
    max_position_pct_per_symbol: float = 0.2

    # Added to satisfy tests/unit/gpt_trader/features/live_trade/test_config_dict_json.py
    min_liquidation_buffer_pct: float = 0.1
    leverage_max_per_symbol: dict[str, int] = field(default_factory=dict)
    max_notional_per_symbol: dict[str, Decimal] = field(default_factory=dict)
    slippage_guard_bps: int = 50
    kill_switch_enabled: bool = False
    reduce_only_mode: bool = False

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            max_leverage=int(os.getenv("MAX_LEVERAGE", "5")),
            daily_loss_limit=Decimal(os.getenv("DAILY_LOSS_LIMIT", "100")),
            max_exposure_pct=float(os.getenv("MAX_EXPOSURE_PCT", "0.8")),
            max_position_pct_per_symbol=float(os.getenv("MAX_POSITION_PCT_PER_SYMBOL", "0.2")),
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
        with open(path, "r") as f:
            data = json.load(f)

        # Type conversion
        if "daily_loss_limit" in data:
            data["daily_loss_limit"] = Decimal(str(data["daily_loss_limit"]))

        if "max_notional_per_symbol" in data:
            data["max_notional_per_symbol"] = {
                k: Decimal(str(v)) for k, v in data["max_notional_per_symbol"].items()
            }

        return cls(**data)


__all__ = ["RiskConfig", "RISK_CONFIG_ENV_KEYS", "RISK_CONFIG_ENV_ALIASES"]
