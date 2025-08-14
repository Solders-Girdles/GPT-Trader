"""
Integration tests for data pipeline.

Tests the complete data flow from sources through validation
and transformation to final consumption.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio

from bot.dataflow.sources.yfinance_source import YFinanceSource
from bot.dataflow.validate import DataFrameValidator
from bot.dataflow.historical_data_manager import HistoricalDataManager
from bot.dataflow.streaming_data import StreamingDataManager
from bot.dataflow.data_quality_framework import DataQualityFramework


class TestDataPipelineIntegration:
    """Integration tests for the data pipeline."""

    @pytest.fixture
    def symbols(self):
        """Test symbols."""
        return ["AAPL", "GOOGL", "MSFT"]

    @pytest.fixture
    def date_range(self):
        """Test date range."""
        return {
            "start": datetime(2023, 1, 1),
            "end": datetime(2023, 12, 31),
        }

    @pytest.fixture
    def mock_raw_data(self):
        """Create mock raw market data."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        
        return pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, len(dates)),
                "High": np.random.uniform(110, 120, len(dates)),
                "Low": np.random.uniform(90, 100, len(dates)),
                "Close": np.random.uniform(95, 115, len(dates)),
                "Volume": np.random.uniform(1e6, 1e8, len(dates)),
                "Adj Close": np.random.uniform(95, 115, len(dates)),
            },
            index=dates
        )

    @pytest.fixture
    def data_source(self):
        """Create data source instance."""
        return YFinanceSource()

    @pytest.fixture
    def data_validator(self):
        """Create data validator instance."""
        return DataValidator()

    @pytest.fixture
    def historical_manager(self):
        """Create historical data manager."""
        return HistoricalDataManager(cache_dir="/tmp/test_cache")

    def test_data_fetching_pipeline(self, data_source, symbols, date_range):
        """Test complete data fetching pipeline."""
        with patch('yfinance.download') as mock_download:
            # Setup mock
            mock_data = pd.DataFrame(
                {
                    "Close": [100, 101, 102],
                    "Volume": [1e6, 1.1e6, 1.2e6],
                },
                index=pd.date_range("2023-01-01", periods=3)
            )
            mock_download.return_value = mock_data
            
            # Fetch data
            data = data_source.fetch_multiple(
                symbols,
                start_date=date_range["start"],
                end_date=date_range["end"]
            )
            
            # Verify structure
            assert isinstance(data, dict)
            assert all(symbol in data for symbol in symbols)
            assert all(isinstance(df, pd.DataFrame) for df in data.values())

    def test_data_validation_pipeline(self, data_validator, mock_raw_data):
        """Test data validation in pipeline."""
        # Test valid data
        is_valid, issues = data_validator.validate(mock_raw_data)
        assert is_valid
        assert len(issues) == 0
        
        # Test data with issues
        bad_data = mock_raw_data.copy()
        bad_data.loc[bad_data.index[10], "Close"] = np.nan
        bad_data.loc[bad_data.index[20], "Volume"] = -1000
        
        is_valid, issues = data_validator.validate(bad_data)
        assert not is_valid
        assert len(issues) > 0
        assert any("NaN" in issue for issue in issues)
        assert any("negative" in issue.lower() for issue in issues)

    def test_data_transformation_pipeline(self, mock_raw_data):
        """Test data transformation steps."""
        # Add technical indicators
        from bot.indicators.atr import calculate_atr
        
        # Transform data
        transformed = mock_raw_data.copy()
        
        # Add moving averages
        transformed["SMA_20"] = transformed["Close"].rolling(20).mean()
        transformed["SMA_50"] = transformed["Close"].rolling(50).mean()
        
        # Add ATR
        transformed["ATR"] = calculate_atr(
            transformed["High"],
            transformed["Low"],
            transformed["Close"],
            period=14
        )
        
        # Verify transformations
        assert "SMA_20" in transformed.columns
        assert "SMA_50" in transformed.columns
        assert "ATR" in transformed.columns
        assert not transformed["ATR"].isna().all()

    def test_data_quality_framework(self, mock_raw_data):
        """Test data quality framework integration."""
        quality_framework = DataQualityFramework()
        
        # Run quality checks
        quality_report = quality_framework.assess_quality(mock_raw_data)
        
        # Verify report structure
        assert "completeness" in quality_report
        assert "accuracy" in quality_report
        assert "consistency" in quality_report
        assert "timeliness" in quality_report
        
        # Check scores
        assert 0 <= quality_report["completeness"] <= 1
        assert 0 <= quality_report["accuracy"] <= 1

    def test_historical_data_management(self, historical_manager, mock_raw_data):
        """Test historical data storage and retrieval."""
        symbol = "AAPL"
        
        # Store data
        historical_manager.store(symbol, mock_raw_data)
        
        # Retrieve data
        retrieved = historical_manager.retrieve(
            symbol,
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        
        # Verify retrieval
        assert retrieved is not None
        assert len(retrieved) == len(mock_raw_data)
        pd.testing.assert_frame_equal(retrieved, mock_raw_data)

    def test_data_gap_handling(self, data_validator):
        """Test handling of data gaps."""
        # Create data with gaps
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        data_with_gaps = pd.DataFrame(
            {
                "Close": [100] * len(dates),
                "Volume": [1e6] * len(dates),
            },
            index=dates
        )
        
        # Remove some dates to create gaps
        data_with_gaps = data_with_gaps.drop(data_with_gaps.index[5:8])
        
        # Detect gaps
        gaps = data_validator.detect_gaps(data_with_gaps)
        
        assert len(gaps) > 0
        assert gaps[0]["start"] == dates[5]
        assert gaps[0]["end"] == dates[7]
        
        # Fill gaps
        filled_data = data_validator.fill_gaps(data_with_gaps, method="forward_fill")
        
        assert len(filled_data) > len(data_with_gaps)
        assert not filled_data["Close"].isna().any()

    def test_real_time_data_streaming(self):
        """Test real-time data streaming integration."""
        streaming_manager = StreamingDataManager()
        
        # Mock streaming data
        async def mock_stream():
            for i in range(5):
                yield {
                    "symbol": "AAPL",
                    "price": 150 + i,
                    "volume": 1000000 + i * 10000,
                    "timestamp": datetime.now() + timedelta(seconds=i)
                }
                await asyncio.sleep(0.1)
        
        # Process stream
        processed_data = []
        
        async def process_stream():
            async for data in mock_stream():
                validated = streaming_manager.validate_tick(data)
                if validated:
                    processed_data.append(validated)
        
        # Run async test
        asyncio.run(process_stream())
        
        assert len(processed_data) == 5
        assert all("symbol" in d for d in processed_data)

    def test_data_aggregation_pipeline(self, mock_raw_data):
        """Test data aggregation across timeframes."""
        # Create minute data
        minute_dates = pd.date_range("2023-01-01", periods=1440, freq="1min")
        minute_data = pd.DataFrame(
            {
                "Close": np.random.uniform(100, 110, len(minute_dates)),
                "Volume": np.random.uniform(100, 1000, len(minute_dates)),
            },
            index=minute_dates
        )
        
        # Aggregate to different timeframes
        aggregations = {
            "5min": minute_data.resample("5min").agg({"Close": "last", "Volume": "sum"}),
            "15min": minute_data.resample("15min").agg({"Close": "last", "Volume": "sum"}),
            "1h": minute_data.resample("1h").agg({"Close": "last", "Volume": "sum"}),
            "1d": minute_data.resample("1d").agg({"Close": "last", "Volume": "sum"}),
        }
        
        # Verify aggregations
        assert len(aggregations["5min"]) == len(minute_data) // 5
        assert len(aggregations["1h"]) == 24
        assert aggregations["1d"]["Volume"].iloc[0] == minute_data["Volume"].sum()

    def test_multi_source_data_fusion(self):
        """Test combining data from multiple sources."""
        # Mock different data sources
        source1_data = pd.DataFrame(
            {
                "price": [100, 101, 102],
                "volume": [1e6, 1.1e6, 1.2e6],
            },
            index=pd.date_range("2023-01-01", periods=3)
        )
        
        source2_data = pd.DataFrame(
            {
                "bid": [99.5, 100.5, 101.5],
                "ask": [100.5, 101.5, 102.5],
            },
            index=pd.date_range("2023-01-01", periods=3)
        )
        
        # Fuse data
        fused_data = pd.concat([source1_data, source2_data], axis=1)
        
        # Add derived features
        fused_data["spread"] = fused_data["ask"] - fused_data["bid"]
        fused_data["mid_price"] = (fused_data["bid"] + fused_data["ask"]) / 2
        
        # Verify fusion
        assert all(col in fused_data.columns for col in ["price", "volume", "bid", "ask", "spread", "mid_price"])
        assert (fused_data["spread"] > 0).all()

    def test_data_normalization(self):
        """Test data normalization in pipeline."""
        # Create data with different scales
        data = pd.DataFrame(
            {
                "price": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1e6, 1e8, 100),
                "volatility": np.random.uniform(0.1, 0.5, 100),
            }
        )
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        # Z-score normalization
        scaler_standard = StandardScaler()
        data_standardized = pd.DataFrame(
            scaler_standard.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        # Min-max normalization
        scaler_minmax = MinMaxScaler()
        data_minmax = pd.DataFrame(
            scaler_minmax.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        # Verify normalization
        assert np.allclose(data_standardized.mean(), 0, atol=1e-7)
        assert np.allclose(data_standardized.std(), 1, atol=1e-7)
        assert (data_minmax >= 0).all().all()
        assert (data_minmax <= 1).all().all()

    def test_data_versioning(self, historical_manager, mock_raw_data):
        """Test data versioning and rollback."""
        symbol = "AAPL"
        
        # Store multiple versions
        v1_data = mock_raw_data.copy()
        v1_data["version"] = 1
        historical_manager.store_versioned(symbol, v1_data, version=1)
        
        v2_data = mock_raw_data.copy()
        v2_data["Close"] = v2_data["Close"] * 1.1
        v2_data["version"] = 2
        historical_manager.store_versioned(symbol, v2_data, version=2)
        
        # Retrieve specific versions
        retrieved_v1 = historical_manager.retrieve_version(symbol, version=1)
        retrieved_v2 = historical_manager.retrieve_version(symbol, version=2)
        
        assert retrieved_v1["version"].iloc[0] == 1
        assert retrieved_v2["version"].iloc[0] == 2
        assert not np.allclose(retrieved_v1["Close"], retrieved_v2["Close"])

    def test_error_recovery_pipeline(self, data_source):
        """Test error recovery in data pipeline."""
        # Simulate failures
        with patch('yfinance.download') as mock_download:
            mock_download.side_effect = [
                Exception("Network error"),
                Exception("API limit"),
                pd.DataFrame({"Close": [100, 101, 102]}),  # Success on third try
            ]
            
            # Fetch with retry
            data = data_source.fetch_with_retry(
                "AAPL",
                max_retries=3,
                retry_delay=0.1
            )
            
            assert data is not None
            assert len(data) == 3

    def test_data_export_formats(self, mock_raw_data, tmp_path):
        """Test exporting data in various formats."""
        # Export to different formats
        formats = {
            "csv": tmp_path / "data.csv",
            "parquet": tmp_path / "data.parquet",
            "json": tmp_path / "data.json",
            "hdf": tmp_path / "data.h5",
        }
        
        # Save in each format
        mock_raw_data.to_csv(formats["csv"])
        mock_raw_data.to_parquet(formats["parquet"])
        mock_raw_data.to_json(formats["json"])
        mock_raw_data.to_hdf(formats["hdf"], key="data")
        
        # Verify all files created
        assert all(path.exists() for path in formats.values())
        
        # Load and verify
        loaded_csv = pd.read_csv(formats["csv"], index_col=0, parse_dates=True)
        loaded_parquet = pd.read_parquet(formats["parquet"])
        
        assert len(loaded_csv) == len(mock_raw_data)
        assert len(loaded_parquet) == len(mock_raw_data)

    @pytest.mark.slow
    def test_large_scale_data_processing(self):
        """Test processing large volumes of data."""
        # Create large dataset
        n_symbols = 100
        n_days = 1000
        
        large_data = {}
        for i in range(n_symbols):
            symbol = f"STOCK_{i}"
            dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
            
            large_data[symbol] = pd.DataFrame(
                {
                    "Close": np.random.uniform(50, 150, n_days),
                    "Volume": np.random.uniform(1e5, 1e7, n_days),
                },
                index=dates
            )
        
        # Process data
        validator = DataValidator()
        
        validation_results = {}
        for symbol, data in large_data.items():
            is_valid, issues = validator.validate(data)
            validation_results[symbol] = is_valid
        
        # All should be valid
        assert all(validation_results.values())
        assert len(validation_results) == n_symbols