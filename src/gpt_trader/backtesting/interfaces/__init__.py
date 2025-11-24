"""Backtesting interfaces for swappable components."""

from .execution import IExecution
from .market_data import IMarketData
from .portfolio import IPortfolio

__all__ = ["IMarketData", "IExecution", "IPortfolio"]
