from __future__ import annotations

from statistics import pstdev
from typing import List

from .interfaces import StrategyBase, StrategyContext, StrategySignal


class VolatilityStrategy(StrategyBase):
    name = "volatility"

    def __init__(self, **params):
        super().__init__(**params)
        self.vol_period = int(params.get("vol_period", 20))
        self.vol_threshold = float(params.get("vol_threshold", 0.02))

    def get_signals(self, ctx: StrategyContext) -> List[StrategySignal]:
        out: List[StrategySignal] = []
        for sym in ctx.symbols:
            prices = self.price_history.get(sym) or []
            if len(prices) < max(self.vol_period + 5, 10):
                continue
            returns = []
            for i in range(1, len(prices)):
                if prices[i - 1] != 0:
                    returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
            if len(returns) < self.vol_period:
                continue
            vol = pstdev(returns[-self.vol_period :]) if len(set(returns[-self.vol_period :])) > 1 else 0.0
            # Low volatility -> trend follow on last 5-step return
            if vol < self.vol_threshold and len(prices) >= 6 and prices[-6] != 0:
                recent = (prices[-1] - prices[-6]) / prices[-6]
                if recent > 0.01:
                    out.append(StrategySignal(symbol=sym, side="buy", confidence=0.8))
                elif recent < -0.01:
                    out.append(StrategySignal(symbol=sym, side="sell", confidence=0.8))
        return out

