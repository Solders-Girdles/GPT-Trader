"""
Compatibility shim for the legacy `bot_v2.data_providers.coinbase_provider` path.

The actual implementation now lives in `bot_v2.data_providers.coinbase`.
"""

from __future__ import annotations

from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.market_data_service import (
    CoinbaseTickerService,
    TickerCache,
)
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.ws import CoinbaseWebSocket

from .coinbase import CoinbaseDataProvider, create_coinbase_provider

__all__ = [
    "CoinbaseDataProvider",
    "create_coinbase_provider",
    "CoinbaseTickerService",
    "TickerCache",
    "CoinbaseWebSocket",
    "CoinbaseClient",
    "CoinbaseBrokerage",
    "APIConfig",
]
