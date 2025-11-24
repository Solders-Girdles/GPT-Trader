from dataclasses import dataclass
from enum import Enum

class DataSource(Enum):
    COINBASE = "coinbase"

class DataType(Enum):
    CANDLE = "candle"

@dataclass
class DataQuery:
    symbol: str
    source: DataSource
    data_type: DataType
