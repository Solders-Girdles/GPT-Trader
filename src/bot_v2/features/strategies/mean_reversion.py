from __future__ import annotations

from statistics import mean, pstdev
from typing import List

from .interfaces import StrategyBase, StrategyContext, StrategySignal


class MeanReversionStrategy(StrategyBase):
    name = "mean_reversion"

    def __init__(self, **params):
        super().__init__(**params)
        self.bb_period = int(params.get("bb_period", 20))
        self.bb_std = float(params.get("bb_std", 2.0))

    def get_signals(self, ctx: StrategyContext) -> List[StrategySignal]:
        out: List[StrategySignal] = []
        for sym in ctx.symbols:
            prices = self.price_history.get(sym) or []
            if len(prices) < self.bb_period:
                continue
            window = prices[-self.bb_period :]
            m = mean(window)
            sd = pstdev(window) if len(set(window)) > 1 else 0.0
            upper = m + self.bb_std * sd
            lower = m - self.bb_std * sd
            curr = prices[-1]
            if curr < lower:
                # confidence based on distance from band
                conf = min(1.0, (lower - curr) / (upper - m) if upper > m else 0.5)
                out.append(StrategySignal(symbol=sym, side="buy", confidence=max(0.2, conf)))
            elif curr > upper:
                conf = min(1.0, (curr - upper) / (upper - m) if upper > m else 0.5)
                out.append(StrategySignal(symbol=sym, side="sell", confidence=max(0.2, conf)))
        return out

