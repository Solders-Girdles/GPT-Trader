"""Compatibility exports for live-trade strategy interfaces."""

from gpt_trader.core import Decision
from gpt_trader.features.live_trade.strategies.base import MarketDataContext, StrategyProtocol

TradingStrategy = StrategyProtocol

__all__ = [
    "Decision",
    "MarketDataContext",
    "TradingStrategy",
]
