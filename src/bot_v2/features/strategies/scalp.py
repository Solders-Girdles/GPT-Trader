from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .interfaces import StrategyBase, StrategyContext, StrategySignal


@dataclass
class ScalpParams:
    bp_threshold: float = 0.0005  # 5 bps = 0.05%


class ScalpStrategy(StrategyBase):
    name = "scalp"

    def __init__(self, **params):
        super().__init__(**params)
        self.bp_threshold = float(params.get("bp_threshold", 0.0005))

    def get_signals(self, ctx: StrategyContext) -> List[StrategySignal]:
        signals: List[StrategySignal] = []
        for sym in ctx.symbols:
            hist = self.price_history.get(sym) or []
            if len(hist) < 2:
                continue
            last = hist[-2]
            if last <= 0:
                continue
            change = (hist[-1] - last) / last
            if change >= self.bp_threshold:
                signals.append(StrategySignal(symbol=sym, side="buy", confidence=min(1.0, change / self.bp_threshold)))
            elif change <= -self.bp_threshold:
                signals.append(StrategySignal(symbol=sym, side="sell", confidence=min(1.0, abs(change) / self.bp_threshold)))
        return signals

