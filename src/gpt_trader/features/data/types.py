"""
Data layer type definitions.

This module defines the core types for the data fetching and caching subsystem:

Types
-----
- ``DataSource``: Enum of supported market data providers (Coinbase, Yahoo)
- ``DataType``: Enum of data formats (OHLCV candles, etc.)
- ``DataQuery``: Request object for fetching historical data

Cache Keys
----------
``DataQuery.get_cache_key()`` generates deterministic cache keys for data queries.
Key format: ``{source}:{data_type}:{symbols}:{start}:{end}``

Example::

    query = DataQuery(
        symbols=["BTC-USD", "ETH-USD"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 31),
        interval="1h",
    )
    cache_key = query.get_cache_key()
"""

from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class DataSource(Enum):
    COINBASE = "coinbase"
    YAHOO = "yahoo"


class DataType(Enum):
    CANDLE = "candle"
    OHLCV = "ohlcv"


@dataclass
class DataQuery:
    symbols: list[str]
    start_date: datetime
    end_date: datetime
    interval: str = "1h"
    data_type: DataType = DataType.OHLCV
    source: DataSource = DataSource.COINBASE

    def get_cache_key(self) -> str:
        # Minimal implementation for tests
        return f"{self.source.value}:{self.data_type.value}:{','.join(sorted(self.symbols))}:{self.start_date.isoformat()}:{self.end_date.isoformat()}"
