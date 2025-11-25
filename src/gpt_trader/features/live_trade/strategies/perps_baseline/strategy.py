from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any

from gpt_trader.features.brokerages.core.interfaces import Product


class Action(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Decision:
    action: Action
    reason: str
    confidence: float = 0.0


@dataclass
class StrategyConfig:
    long_ma: int = 20
    short_ma: int = 5
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.05
    max_leverage: int = 5
    kill_switch_enabled: bool = False


class BaselinePerpsStrategy:
    def __init__(self, config: Any = None):
        self.config = config

    def decide(
        self,
        symbol: str,
        current_mark: Decimal,
        position_state: dict[str, Any] | None,
        recent_marks: Sequence[Decimal],
        equity: Decimal,
        product: Product | None,
    ) -> Decision:
        if not recent_marks:
            return Decision(Action.HOLD, "Insufficient data")

        # Simple Logic:
        # If current > avg of recent -> BUY
        # If current < avg of recent -> SELL

        avg = sum(recent_marks) / len(recent_marks)

        if current_mark > avg:
            return Decision(Action.BUY, f"Price {current_mark:.2f} > Avg {avg:.2f}", 0.6)
        elif current_mark < avg:
            return Decision(Action.SELL, f"Price {current_mark:.2f} < Avg {avg:.2f}", 0.6)

        return Decision(Action.HOLD, "No signal")

    def _build_default_product(self, symbol: str) -> Any:
        return None
