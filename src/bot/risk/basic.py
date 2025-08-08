from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskParams:
    risk_per_trade_usd: float = 1000.0
    stop_pct: float = 0.02
    take_pct: float = 0.04


# Placeholder for later expansion
