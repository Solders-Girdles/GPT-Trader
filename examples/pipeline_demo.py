#!/usr/bin/env python3
"""Demo script for the unified data pipeline."""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.dataflow.pipeline import (
    DataPipeline,
    PipelineConfig,
    MultiSourceConfig,
    DataSourceConfig,
    DataSourceType,
)
from bot.logging import get_logger

logger = get_logger("demo")


def demo_basic_pipeline():
    """Demonstrate basic pipeline usage with default YFinance source."""
    print("\n=== Basic Pipeline Demo ===")

    # Create pipeline with default configuration
    pipeline = DataPipeline()

    # Fetch data for a few symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)

    print(f"Fetching data for {symbols}...")

    try:
        data = pipeline.fetch_and_validate(symbols, start_date, end_date)

        print(f"Successfully loaded data for {len(data)} symbols:")
        for symbol, df in data.items():
            print(f"  {symbol}: {len(df)} data points, latest close: ${df['Close'].iloc[-1]:.2f}")

        # Show metrics
        metrics = pipeline.get_metrics()
        print(f"\nMetrics: {metrics.to_dict()}")

    except Exception as e:
        print(f"Error: {e}")


def demo_multi_source_pipeline():
    """Demonstrate multi-source pipeline with failover."""
    print("\n=== Multi-Source Pipeline Demo ===")

    # Configure multiple data sources
    multi_config = MultiSourceConfig(
        sources=[
            DataSourceConfig(
                source_type=DataSourceType.YFINANCE,
                priority=1,  # Primary source
                timeout_seconds=10.0,
            ),
            DataSourceConfig(
                source_type=DataSourceType.ENHANCED_YFINANCE,
                priority=2,  # Fallback source
                timeout_seconds=20.0,
            ),
        ],
        failover_enabled=True,
    )

    # Create pipeline with multi-source config
    pipeline = DataPipeline(multi_source_config=multi_config)

    # Show source information
    source_info = pipeline.get_source_info()
    print(f"Configured sources: {source_info}")

    # Fetch data
    symbols = ["AAPL"]
    start_date = datetime(2023, 6, 1)
    end_date = datetime(2023, 6, 30)

    try:
        data = pipeline.fetch_and_validate(symbols, start_date, end_date)

        for symbol, df in data.items():
            print(f"Successfully fetched {len(df)} data points for {symbol}")
            print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

    except Exception as e:
        print(f"Error: {e}")


def demo_csv_source_pipeline():
    """Demonstrate pipeline with CSV file source."""
    print("\n=== CSV Source Pipeline Demo ===")

    # Create a sample CSV file for demonstration
    sample_csv_path = "/tmp/sample_market_data.csv"
    create_sample_csv(sample_csv_path)

    # Configure CSV source
    multi_config = MultiSourceConfig(
        sources=[
            DataSourceConfig(
                source_type=DataSourceType.CSV_FILE,
                priority=1,
                config_params={"file_path": sample_csv_path, "date_column": "Date"},
            )
        ]
    )

    pipeline = DataPipeline(multi_source_config=multi_config)

    # Fetch data from CSV
    try:
        data = pipeline.fetch_and_validate(["SAMPLE"], datetime(2023, 1, 1), datetime(2023, 1, 10))

        for symbol, df in data.items():
            print(f"Loaded {len(df)} rows from CSV for {symbol}")
            print(df.head())

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Clean up
        if os.path.exists(sample_csv_path):
            os.remove(sample_csv_path)


def create_sample_csv(file_path: str):
    """Create a sample CSV file for demonstration."""
    import pandas as pd

    # Generate sample data
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    data = {
        "Date": dates,
        "Open": [100.0 + i for i in range(10)],
        "High": [105.0 + i for i in range(10)],
        "Low": [95.0 + i for i in range(10)],
        "Close": [102.0 + i for i in range(10)],
        "Volume": [1000 + i * 100 for i in range(10)],
        "Symbol": ["SAMPLE"] * 10,
    }

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Created sample CSV file: {file_path}")


def demo_caching_performance():
    """Demonstrate caching performance benefits."""
    print("\n=== Caching Performance Demo ===")

    pipeline = DataPipeline()
    symbol = "AAPL"
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)

    # First fetch (cold cache)
    print("First fetch (cache miss):")
    start_time = datetime.now()
    data1 = pipeline.fetch_and_validate([symbol], start_date, end_date)
    time1 = (datetime.now() - start_time).total_seconds() * 1000
    print(f"Time: {time1:.2f}ms")

    # Second fetch (warm cache)
    print("\nSecond fetch (cache hit):")
    start_time = datetime.now()
    data2 = pipeline.fetch_and_validate([symbol], start_date, end_date)
    time2 = (datetime.now() - start_time).total_seconds() * 1000
    print(f"Time: {time2:.2f}ms")

    print(f"\nSpeedup: {time1/time2:.1f}x faster")

    # Show cache info
    cache_info = pipeline.get_cache_info()
    print(f"Cache info: {cache_info}")


def demo_pipeline_health_check():
    """Demonstrate pipeline health monitoring."""
    print("\n=== Pipeline Health Check Demo ===")

    pipeline = DataPipeline()

    health = pipeline.health_check("AAPL")

    print(f"Pipeline Status: {health['status']}")
    print(f"Errors: {len(health['errors'])}")
    print(f"Warnings: {len(health['warnings'])}")

    for test_name, test_result in health["tests"].items():
        status = "PASS" if test_result.get("success", False) else "FAIL"
        print(f"  {test_name}: {status}")

        if "response_time_ms" in test_result:
            print(f"    Response time: {test_result['response_time_ms']:.2f}ms")

        if "data_points" in test_result:
            print(f"    Data points: {test_result['data_points']}")


def demo_error_handling():
    """Demonstrate error handling and graceful degradation."""
    print("\n=== Error Handling Demo ===")

    # Configure strict validation
    strict_config = PipelineConfig(strict_validation=True, fail_on_missing_symbols=True)

    # Configure lenient validation
    lenient_config = PipelineConfig(strict_validation=False, fail_on_missing_symbols=False)

    symbols = ["AAPL", "INVALID_SYMBOL_123"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 31)

    # Test strict mode
    print("Testing strict mode (should handle errors gracefully):")
    strict_pipeline = DataPipeline(strict_config)
    try:
        data = strict_pipeline.fetch_and_validate(symbols, start_date, end_date)
        print(f"Loaded data for {len(data)} symbols in strict mode")
    except Exception as e:
        print(f"Strict mode error (expected): {e}")

    # Test lenient mode
    print("\nTesting lenient mode:")
    lenient_pipeline = DataPipeline(lenient_config)
    try:
        data = lenient_pipeline.fetch_and_validate(symbols, start_date, end_date)
        print(f"Loaded data for {len(data)} symbols in lenient mode")

        metrics = lenient_pipeline.get_metrics()
        print(f"Success rate: {metrics.success_rate:.1f}%")
        print(f"Failed symbols: {metrics.symbols_failed}")
    except Exception as e:
        print(f"Lenient mode error: {e}")


def main():
    """Run all demo functions."""
    print("GPT-Trader Data Pipeline Demo")
    print("=============================")

    demos = [
        demo_basic_pipeline,
        demo_multi_source_pipeline,
        demo_csv_source_pipeline,
        demo_caching_performance,
        demo_pipeline_health_check,
        demo_error_handling,
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"Demo {demo.__name__} failed: {e}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
