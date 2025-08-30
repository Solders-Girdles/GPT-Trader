"""
Paper trading feature slice - simulated live trading with real-time data.

Complete isolation - no external dependencies.
"""

from .paper_trade import start_paper_trading, stop_paper_trading, get_status
from .types import PaperTradeResult, Position, TradeLog

__all__ = [
    'start_paper_trading',
    'stop_paper_trading', 
    'get_status',
    'PaperTradeResult',
    'Position',
    'TradeLog'
]