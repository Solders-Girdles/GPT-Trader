# Data Feature

**Purpose**: Data management, caching, quality control, and storage abstraction.

---

## Overview

The `data` feature provides:
- Market data retrieval and caching
- Data quality validation and cleaning
- Storage abstraction (file, database, memory)
- Historical data management

**Coverage**: 🟢 96.4% (Excellent)

---

## Interface Contract

### Inputs

#### Required Dependencies
```python
from bot_v2.features.data import (
    DataCache,
    DataQualityChecker,
    DataStorage
)
from datetime import datetime
```

#### Configuration
- **Cache TTL**: Time-to-live for cached data
- **Storage Backend**: file, redis, or memory
- **Quality Thresholds**: Min data points, max gap duration

### Outputs

#### Data Structures
```python
from bot_v2.features.data.types import (
    CachedData,
    QualityReport,
    StorageMetadata
)
```

#### Return Values
- **Cached Data**: Time-stamped data with TTL metadata
- **Quality Report**: Validation results with issue details
- **Storage Status**: Available space, latency metrics

### Side Effects

#### State Modifications
- ✅ Updates cache entries
- ✅ Writes to configured storage backend
- ✅ Cleans expired cache entries

#### External Interactions
- 💾 Reads/writes from/to file system or Redis
- 📊 Emits cache hit/miss metrics

---

## Core Modules

### Data Cache (`cache.py`)
```python
class DataCache:
    """In-memory cache with TTL support."""

    def get(self, key: str) -> Any | None:
        """Get cached value if not expired."""

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Cache value with TTL."""

    def clear_expired(self) -> int:
        """Remove expired entries, return count cleared."""
```

### Data Quality (`quality.py`)
```python
def validate_ohlcv(df: pd.DataFrame) -> QualityReport:
    """Validate OHLCV data quality."""

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers and fill gaps."""

def detect_anomalies(series: pd.Series) -> list[int]:
    """Detect anomalous data points."""
```

### Data Storage (`storage.py`)
```python
class DataStorage:
    """Abstract storage interface."""

    def save(self, key: str, data: bytes, metadata: dict) -> None:
        """Save data with metadata."""

    def load(self, key: str) -> tuple[bytes, dict]:
        """Load data and metadata."""

    def list_keys(self, prefix: str = "") -> list[str]:
        """List stored keys."""
```

---

## Usage Examples

### Basic Caching
```python
from bot_v2.features.data import DataCache

cache = DataCache(max_size=1000)

# Cache price data for 5 minutes
cache.set("BTC-USD:price", {"price": 50000, "timestamp": "..."}, ttl_seconds=300)

# Retrieve cached data
price_data = cache.get("BTC-USD:price")
if price_data:
    print(f"Cached price: {price_data['price']}")
else:
    # Fetch fresh data
    price_data = fetch_from_api()
    cache.set("BTC-USD:price", price_data, ttl_seconds=300)
```

### Data Quality Validation
```python
from bot_v2.features.data import validate_ohlcv
import pandas as pd

# Load OHLCV data
df = pd.read_csv("prices.csv")

# Validate quality
report = validate_ohlcv(df)

if not report.is_valid:
    print(f"Quality issues found:")
    for issue in report.issues:
        print(f"  - {issue.severity}: {issue.message}")

# Clean data
from bot_v2.features.data import clean_price_data
clean_df = clean_price_data(df)
```

### Storage Operations
```python
from bot_v2.features.data import DataStorage
import pickle

storage = DataStorage(backend="file", base_path="./data")

# Save market data
data = {"symbol": "BTC-USD", "prices": [...]}
storage.save(
    "market/BTC-USD/2024-10",
    pickle.dumps(data),
    metadata={"format": "pickle", "version": 1}
)

# Load data
data_bytes, metadata = storage.load("market/BTC-USD/2024-10")
data = pickle.loads(data_bytes)
```

---

## Configuration

### Cache Settings
```python
# In bot config or env
CACHE_MAX_SIZE = 10000          # Max cached items
CACHE_DEFAULT_TTL = 300         # Default TTL in seconds
CACHE_CLEANUP_INTERVAL = 60     # Cleanup frequency
```

### Storage Settings
```python
STORAGE_BACKEND = "file"        # file, redis, memory
STORAGE_BASE_PATH = "./data"    # For file backend
STORAGE_REDIS_URL = "..."       # For redis backend
```

---

## Testing Strategy

### Unit Tests (`tests/unit/bot_v2/features/data/`)
- Cache TTL expiration tests
- Quality validation edge cases
- Storage backend switching

### Integration Tests
- Redis cache integration (if available)
- File I/O with permissions tests
- Concurrent cache access

---

## Performance Characteristics

### Cache Performance
- **Get**: O(1) average
- **Set**: O(1) average
- **Cleanup**: O(n) where n = expired entries

### Storage Performance
- **File Backend**: I/O bound, ~100 ops/sec
- **Redis Backend**: Network bound, ~10k ops/sec
- **Memory Backend**: CPU bound, ~1M ops/sec

---

## Data Quality Checks

### OHLCV Validation
- ✅ No missing OHLC columns
- ✅ Prices are positive
- ✅ High ≥ Low always
- ✅ Close within [Low, High]
- ✅ Volume ≥ 0
- ✅ Timestamps are monotonic
- ✅ No extreme gaps in time series

### Anomaly Detection
- Outlier detection (3σ from mean)
- Sudden price jumps (>5% in 1 bar)
- Volume spikes (>10x average)

---

## Dependencies

### Internal
- `bot_v2.shared.types` - Type definitions

### External
- `pandas` - Data manipulation
- `redis` (optional) - Redis cache backend

---

**Last Updated**: 2025-10-05
**Status**: ✅ Production (Stable)
