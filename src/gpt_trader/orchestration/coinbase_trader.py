"""Neutral entry point for the Coinbase Trader runtime.

The implementation lives in :mod:`gpt_trader.orchestration.trading_bot`. This module provides a
spot-first alias while keeping the legacy ``PerpsBot`` name importable for backwards compatibility.
"""

from __future__ import annotations

from gpt_trader.orchestration.trading_bot import TradingBot

CoinbaseTrader = TradingBot
PerpsBot = TradingBot

__all__ = ["CoinbaseTrader", "PerpsBot", "TradingBot"]
