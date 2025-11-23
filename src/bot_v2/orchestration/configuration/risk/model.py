"""
Simplified Risk Config.
"""
from dataclasses import dataclass, field
from decimal import Decimal
import os

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

__all__ = ["RiskConfig"]
