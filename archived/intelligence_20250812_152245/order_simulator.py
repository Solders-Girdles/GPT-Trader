from __future__ import annotations

from dataclasses import dataclass

from bot.intelligence.transition_metrics import SlippageModel


@dataclass
class OrderSimulationResult:
    executed_price: float
    slippage: float
    execution_time: float
    fill_ratio: float
    market_impact: float


class L2SlippageModel(SlippageModel):
    """Simple L2 slippage model for testing transition costs."""

    def __init__(self, base_slippage: float = 0.0001, impact_factor: float = 0.1) -> None:
        self.base_slippage = float(base_slippage)
        self.impact_factor = float(impact_factor)

    def estimate_slippage(self, trade_size: float, market_volume: float = 1e6) -> float:
        slippage = self.base_slippage
        volume_ratio = float(trade_size) / float(market_volume) if market_volume else 0.0
        market_impact = self.impact_factor * volume_ratio
        return float(slippage + market_impact)

    def simulate_order(
        self, order_size: float, market_price: float, market_volume: float = 1e6
    ) -> OrderSimulationResult:
        slippage_rate = self.estimate_slippage(order_size, market_volume)
        executed_price = float(market_price) * (1.0 + float(slippage_rate))
        slippage = executed_price - float(market_price)
        return OrderSimulationResult(
            executed_price=executed_price,
            slippage=slippage,
            execution_time=1.0,
            fill_ratio=1.0,
            market_impact=slippage_rate,
        )
