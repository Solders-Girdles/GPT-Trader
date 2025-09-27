"""
Local types for data management.

Complete isolation - no external dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import pandas as pd


class DataSource(Enum):
    """Data source enumeration."""
    YAHOO = "yahoo"
    ALPACA = "alpaca"
    POLYGON = "polygon"
    CSV = "csv"
    DATABASE = "database"
    API = "api"


class DataType(Enum):
    """Type of market data."""
    OHLCV = "ohlcv"
    QUOTE = "quote"
    TRADE = "trade"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    INDICATOR = "indicator"


@dataclass
class DataRecord:
    """Single data record."""
    symbol: str
    timestamp: datetime
    data_type: DataType
    source: DataSource
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'data_type': self.data_type.value,
            'source': self.source.value,
            'data': self.data,
            'metadata': self.metadata
        }


@dataclass
class DataQuery:
    """Query for fetching data."""
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    data_type: DataType = DataType.OHLCV
    source: Optional[DataSource] = None
    interval: str = "1d"
    filters: Optional[Dict[str, Any]] = None
    
    def get_cache_key(self) -> str:
        """Generate cache key for this query."""
        symbols_str = ",".join(sorted(self.symbols))
        key = f"{symbols_str}:{self.start_date.date()}:{self.end_date.date()}:{self.data_type.value}:{self.interval}"
        if self.source:
            key += f":{self.source.value}"
        return key


@dataclass
class CacheEntry:
    """Cache entry for stored data."""
    key: str
    data: pd.DataFrame
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > self.expires_at
    
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class StorageStats:
    """Storage statistics."""
    total_records: int
    total_size_mb: float
    oldest_record: Optional[datetime]
    newest_record: Optional[datetime]
    symbols_count: int
    cache_entries: int
    cache_size_mb: float
    cache_hit_rate: float
    
    def summary(self) -> str:
        """Generate storage summary."""
        return f"""
Data Storage Summary
====================
Total Records: {self.total_records:,}
Total Size: {self.total_size_mb:.2f} MB
Date Range: {self.oldest_record.date() if self.oldest_record else 'N/A'} to {self.newest_record.date() if self.newest_record else 'N/A'}
Symbols: {self.symbols_count}

Cache Statistics:
- Entries: {self.cache_entries}
- Size: {self.cache_size_mb:.2f} MB
- Hit Rate: {self.cache_hit_rate:.2%}
        """.strip()


@dataclass
class DataQuality:
    """Data quality metrics."""
    completeness: float  # Percentage of non-null values
    accuracy: float      # Percentage of valid values
    consistency: float   # Percentage of consistent values
    timeliness: float   # How recent the data is
    
    def overall_score(self) -> float:
        """Calculate overall quality score."""
        return (self.completeness + self.accuracy + 
                self.consistency + self.timeliness) / 4
    
    def is_acceptable(self, threshold: float = 0.8) -> bool:
        """Check if quality meets threshold."""
        return self.overall_score() >= threshold


@dataclass
class DataUpdate:
    """Data update notification."""
    symbol: str
    data_type: DataType
    timestamp: datetime
    records_added: int
    records_updated: int
    source: DataSource