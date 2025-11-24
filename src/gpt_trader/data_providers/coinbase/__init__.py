"""
Modular Coinbase data provider package.

Provides the `CoinbaseDataProvider` and factory helpers while allowing the
legacy import path `gpt_trader.data_providers.coinbase_provider` to keep working.
"""

from __future__ import annotations

from .factory import create_coinbase_provider
from .provider import CoinbaseDataProvider

__all__ = ["CoinbaseDataProvider", "create_coinbase_provider"]
