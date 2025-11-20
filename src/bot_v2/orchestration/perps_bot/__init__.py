"""Perps bot orchestration package."""

from .bot import PerpsBot
from .symbol_processing import _CallableSymbolProcessor

# Alias for compatibility if needed, though we are cleaning up
CoinbaseTrader = PerpsBot

__all__ = ["PerpsBot", "CoinbaseTrader", "_CallableSymbolProcessor"]
