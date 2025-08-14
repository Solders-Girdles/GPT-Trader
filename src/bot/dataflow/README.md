# Data Pipeline Documentation

The unified data pipeline provides a robust, flexible system for fetching, validating, and caching market data for backtesting and live trading.

## Features

- **Multi-source support**: YFinance, Enhanced YFinance, CSV files, and extensible to other sources
- **Intelligent caching**: TTL-based caching with automatic invalidation
- **Data validation**: Comprehensive quality checks and error handling
- **Failover mechanism**: Automatic fallback to backup data sources
- **Quality metrics**: Detailed monitoring of pipeline performance
- **Health monitoring**: Built-in diagnostics and health checks
- **Configurable behavior**: Flexible configuration for different use cases

## Quick Start

```python
from bot.dataflow.pipeline import DataPipeline
from datetime import datetime

# Create pipeline with default configuration
pipeline = DataPipeline()

# Fetch data for multiple symbols
symbols = ['AAPL', 'GOOGL', 'MSFT']
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)

data = pipeline.fetch_and_validate(symbols, start_date, end_date)

# Access data for each symbol
for symbol, df in data.items():
    print(f"{symbol}: {len(df)} data points")
    print(f"Latest close: ${df['Close'].iloc[-1]:.2f}")
```

## Configuration

### Basic Pipeline Configuration

```python
from bot.dataflow.pipeline import DataPipeline, PipelineConfig

config = PipelineConfig(
    use_cache=True,
    cache_ttl_hours=24,
    strict_validation=True,
    min_data_points=10,
    apply_adjustments=True,
    fail_on_missing_symbols=False
)

pipeline = DataPipeline(config)
```

### Multi-Source Configuration

```python
from bot.dataflow.pipeline import (
    DataPipeline, MultiSourceConfig, DataSourceConfig, DataSourceType
)

# Configure multiple data sources with priorities
multi_config = MultiSourceConfig(
    sources=[
        DataSourceConfig(
            source_type=DataSourceType.YFINANCE,
            priority=1,  # Primary source
            timeout_seconds=10.0
        ),
        DataSourceConfig(
            source_type=DataSourceType.ENHANCED_YFINANCE,
            priority=2,  # Fallback source
            timeout_seconds=20.0
        )
    ],
    failover_enabled=True
)

pipeline = DataPipeline(multi_source_config=multi_config)
```

### CSV File Source

```python
# Configure CSV file as data source
csv_config = MultiSourceConfig(
    sources=[
        DataSourceConfig(
            source_type=DataSourceType.CSV_FILE,
            priority=1,
            config_params={
                'file_path': '/path/to/market_data.csv',
                'date_column': 'Date'
            }
        )
    ]
)

pipeline = DataPipeline(multi_source_config=csv_config)
```

## Advanced Usage

### Adding Custom Data Sources

```python
# Create custom data source (must implement DataSourceProtocol)
class CustomDataSource:
    def get_daily_bars(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        # Your custom implementation
        pass

# Add to pipeline
custom_source = CustomDataSource()
pipeline.add_data_source(DataSourceType.CSV_FILE, custom_source, priority=1)
```

### Cache Management

```python
# Warm cache for frequently used symbols
symbols = ['AAPL', 'GOOGL', 'MSFT']
results = pipeline.warm_cache(symbols, start_date, end_date)

# Check cache status
cache_info = pipeline.get_cache_info()
print(f"Cache entries: {cache_info['total_entries']}")
print(f"Memory usage: {cache_info['estimated_memory_mb']:.2f} MB")

# Clear cache
pipeline.clear_cache()  # Clear all
pipeline.clear_cache('AAPL')  # Clear specific symbol
```

### Quality Monitoring

```python
# Get pipeline metrics
metrics = pipeline.get_metrics()
print(f"Success rate: {metrics.success_rate:.1f}%")
print(f"Cache hit rate: {metrics.cache_hit_rate:.1f}%")
print(f"Validation errors: {metrics.validation_errors}")

# Health check
health = pipeline.health_check('AAPL')
print(f"Status: {health['status']}")
if health['errors']:
    print("Errors:", health['errors'])
```

### Source Information

```python
# Get information about configured sources
source_info = pipeline.get_source_info()
print(f"Total sources: {source_info['total_sources']}")
print(f"Available sources: {source_info['available_sources']}")
print(f"Primary source: {source_info['primary_source']}")

for source in source_info['sources']:
    print(f"- {source['type']}: priority={source['priority']}, enabled={source['enabled']}")
```

## Error Handling

The pipeline provides several levels of error handling:

### Strict vs Lenient Mode

```python
# Strict mode: fail fast on validation errors
strict_config = PipelineConfig(
    strict_validation=True,
    fail_on_missing_symbols=True
)

# Lenient mode: continue with warnings
lenient_config = PipelineConfig(
    strict_validation=False,
    fail_on_missing_symbols=False
)
```

### Handling Missing Data

```python
try:
    data = pipeline.fetch_and_validate(symbols, start_date, end_date)
except ValueError as e:
    print(f"Pipeline error: {e}")

    # Check what went wrong
    metrics = pipeline.get_metrics()
    if metrics.symbols_failed > 0:
        print(f"Failed symbols: {metrics.symbols_failed}")
        print("Errors:", metrics.errors)
```

## Data Quality Checks

The pipeline performs several quality checks:

- **Minimum data points**: Ensures sufficient data for analysis
- **Missing values**: Detects excessive NaN values in price data
- **Price validation**: Checks for non-positive or unrealistic prices
- **Date consistency**: Validates date ranges and detects gaps
- **Schema validation**: Ensures required columns are present

## Performance Tips

1. **Use caching**: Enable caching for frequently accessed data
2. **Warm cache**: Pre-load data for symbols you'll use repeatedly
3. **Configure timeouts**: Set appropriate timeouts for your network conditions
4. **Monitor metrics**: Use quality metrics to identify performance issues
5. **Choose sources wisely**: Configure failover sources based on reliability

## CSV File Format

When using CSV files as data sources, ensure they follow this format:

```csv
Date,Open,High,Low,Close,Volume,Symbol
2023-01-01,100.0,105.0,99.0,102.0,1000000,AAPL
2023-01-02,102.0,106.0,101.0,104.0,1100000,AAPL
...
```

- **Date**: Date column (configurable name)
- **Open, High, Low, Close**: Required price columns
- **Volume**: Optional volume column
- **Symbol**: Optional symbol column (for multi-symbol files)

## Testing

Run the test suite:

```bash
# Run all pipeline tests
pytest tests/unit/dataflow/ -v

# Run specific test files
pytest tests/unit/dataflow/test_pipeline.py -v
pytest tests/unit/dataflow/test_pipeline_multisource.py -v
```

## Examples

See the demo script for comprehensive examples:

```bash
python examples/pipeline_demo.py
```

## Architecture

```
DataPipeline
├── Configuration
│   ├── PipelineConfig (caching, validation settings)
│   └── MultiSourceConfig (source priorities, failover)
├── Data Sources
│   ├── YFinanceSource (primary)
│   ├── EnhancedYFinanceSource (extended features)
│   ├── CSVFileSource (file-based)
│   └── Custom sources (extensible)
├── Caching System
│   ├── TTL-based cache
│   ├── Memory usage tracking
│   └── Cache warming/clearing
├── Validation Framework
│   ├── Schema validation
│   ├── Data quality checks
│   └── Error handling
└── Monitoring
    ├── Quality metrics
    ├── Health checks
    └── Performance tracking
```

## Integration with GPT-Trader

The pipeline integrates seamlessly with other GPT-Trader components:

- **Backtesting**: Provides validated historical data
- **Strategy development**: Ensures data quality for signal generation
- **Risk management**: Supplies reliable data for risk calculations
- **Portfolio management**: Delivers consistent data for allocation decisions

## Contributing

When adding new data sources:

1. Implement the `DataSourceProtocol` interface
2. Add appropriate configuration options
3. Include comprehensive tests
4. Update documentation
5. Add to the `DataSourceType` enum if needed

For feature requests or bug reports, please follow the project's contribution guidelines.
