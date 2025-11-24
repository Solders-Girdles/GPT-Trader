"""
Simplified Risk Config.
"""
from dataclasses import dataclass, field
from decimal import Decimal
import os

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

    @classmethod
    def from_env(cls) -> "RiskConfig":
        return cls(
            max_leverage=int(os.getenv("MAX_LEVERAGE", "5")),
            daily_loss_limit=Decimal(os.getenv("DAILY_LOSS_LIMIT", "100")),
            max_exposure_pct=float(os.getenv("MAX_EXPOSURE_PCT", "0.8")),
            max_position_pct_per_symbol=float(os.getenv("MAX_POSITION_PCT_PER_SYMBOL", "0.2")),
        )

__all__ = ["RiskConfig", "RISK_CONFIG_ENV_KEYS", "RISK_CONFIG_ENV_ALIASES"]
