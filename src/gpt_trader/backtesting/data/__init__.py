"""Historical data management for backtesting."""

from .fetcher import CoinbaseHistoricalFetcher
from .manager import (
    HistoricalDataManager,
    HistoricalDataUnavailableError,
    create_coinbase_data_provider,
    create_offline_data_provider,
)

__all__ = [
    "CoinbaseHistoricalFetcher",
    "HistoricalDataManager",
    "HistoricalDataUnavailableError",
    "create_coinbase_data_provider",
    "create_offline_data_provider",
]
