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
