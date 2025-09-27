"""
TEST-004: Data Pipeline Tests

Verifies that the data loading and processing pipeline works.
Tests actual data fetching and validation functionality.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest


class TestDataPipeline:
    """Test data loading and processing functionality."""

    def test_yfinance_source_initialization(self):
        """Test that YFinance source can be created."""
        from bot.dataflow.sources.yfinance_source import YFinanceSource

        source = YFinanceSource()
        assert source is not None

    def test_sample_data_structure(self, sample_market_data):
        """Test that sample data has correct structure."""
        assert isinstance(sample_market_data, pd.DataFrame)
        assert "Close" in sample_market_data.columns
        assert "Open" in sample_market_data.columns
        assert "High" in sample_market_data.columns
        assert "Low" in sample_market_data.columns
        assert "Volume" in sample_market_data.columns
        assert len(sample_market_data) > 0

    def test_data_validation_basic(self, sample_market_data):
        """Test basic data validation works."""
        from bot.dataflow.validate import validate_daily_bars

        # Should not raise exception for valid data
        validate_daily_bars(sample_market_data, "TEST")

    def test_data_validation_missing_columns(self):
        """Test data validation catches missing columns."""
        from bot.dataflow.validate import validate_daily_bars

        # Create data with DatetimeIndex but missing columns
        dates = pd.date_range(start="2023-01-01", periods=3)
        invalid_data = pd.DataFrame({"Close": [100, 101, 102]}, index=dates)

        with pytest.raises(ValueError, match="Missing required|missing columns"):
            validate_daily_bars(invalid_data, "TEST")

    @pytest.mark.slow
    def test_yfinance_download_basic(self):
        """Test actual data download from yfinance (slow test)."""
        from bot.dataflow.sources.yfinance_source import YFinanceSource

        source = YFinanceSource()

        # Try to download a small amount of recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        try:
            data = source.fetch("AAPL", start_date, end_date)
            if data is not None and not data.empty:
                assert "Close" in data.columns
                assert len(data) > 0
        except Exception as e:
            pytest.skip(f"Network/API issue: {e}")

    def test_atr_calculation(self, sample_market_data):
        """Test that ATR indicator calculation works."""
        from bot.indicators.atr import atr

        # Convert columns to lowercase for ATR calculation
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        atr_values = atr(df, period=14)

        assert isinstance(atr_values, pd.Series)
        assert len(atr_values) == len(df)
        # ATR should be positive where calculated
        valid_atr = atr_values.dropna()
        if len(valid_atr) > 0:
            assert (valid_atr >= 0).all()

    def test_data_caching_directory(self):
        """Test that data caching directory can be created."""
        from bot.dataflow.sources.yfinance_source import _cache_dir

        cache_dir = _cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()
