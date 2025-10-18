"""Data adapters and abstractions."""

from .base import MarketData
from .yahoo import YahooMarketData

__all__ = ["MarketData", "YahooMarketData"]
