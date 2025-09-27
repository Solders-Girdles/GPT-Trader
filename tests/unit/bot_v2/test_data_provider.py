"""
Unit tests for the data provider abstraction layer
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from bot_v2.data_providers import (
    DataProvider,
    YFinanceProvider,
    MockProvider,
    get_data_provider,
    set_data_provider
)


class TestMockProvider:
    """Test the mock data provider"""
    
    def test_mock_provider_initialization(self):
        """Test that mock provider initializes correctly"""
        provider = MockProvider()
        assert isinstance(provider, DataProvider)
    
    def test_get_historical_data(self):
        """Test getting historical data"""
        provider = MockProvider()
        data = provider.get_historical_data("AAPL", "30d")
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 30
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        assert isinstance(data.index, pd.DatetimeIndex)
    
    def test_get_current_price(self):
        """Test getting current price"""
        provider = MockProvider()
        price = provider.get_current_price("AAPL")
        
        assert isinstance(price, float)
        assert price > 0
        assert price == 150.0  # Mock should return consistent price
    
    def test_get_multiple_symbols(self):
        """Test getting data for multiple symbols"""
        provider = MockProvider()
        symbols = ["AAPL", "GOOGL", "MSFT"]
        data = provider.get_multiple_symbols(symbols, "20d")
        
        assert isinstance(data, dict)
        assert len(data) == 3
        for symbol in symbols:
            assert symbol in data
            assert isinstance(data[symbol], pd.DataFrame)
            assert len(data[symbol]) == 20
    
    def test_is_market_open(self):
        """Test market open check"""
        provider = MockProvider()
        is_open = provider.is_market_open()
        
        assert isinstance(is_open, bool)
        assert is_open is True  # Mock always returns True
    
    def test_deterministic_data(self):
        """Test that mock data is deterministic for same symbol"""
        provider = MockProvider()
        data1 = provider.get_historical_data("AAPL", "10d")
        data2 = provider.get_historical_data("AAPL", "10d")
        
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_different_symbols_different_data(self):
        """Test that different symbols produce different data"""
        provider = MockProvider()
        aapl_data = provider.get_historical_data("AAPL", "10d")
        googl_data = provider.get_historical_data("GOOGL", "10d")
        
        # Data should be different but same structure
        assert not aapl_data['Close'].equals(googl_data['Close'])
        assert aapl_data.shape == googl_data.shape


class TestYFinanceProvider:
    """Test the YFinance provider (with mocking to avoid API calls)"""
    
    def test_yfinance_provider_initialization(self):
        """Test that YFinance provider initializes correctly"""
        provider = YFinanceProvider()
        assert isinstance(provider, DataProvider)
        assert provider._cache == {}
        assert provider._cache_expiry == {}
    
    def test_cache_mechanism(self):
        """Test that caching works properly"""
        provider = YFinanceProvider()
        
        # Mock the yfinance calls
        provider._get_mock_data = lambda symbol, period: pd.DataFrame({
            'Open': [100], 'High': [105], 'Low': [95], 
            'Close': [102], 'Volume': [1000000]
        }, index=[datetime.now()])
        
        # First call should populate cache
        data1 = provider._get_mock_data("AAPL", "1d")
        cache_key = "AAPL_1d_1d"
        provider._cache[cache_key] = data1
        provider._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)
        
        # Second call should use cache
        assert provider._is_cache_valid(cache_key)
    
    def test_fallback_to_mock_data(self):
        """Test fallback to mock data when yfinance fails"""
        provider = YFinanceProvider()
        
        # Force fallback by not having yfinance installed
        data = provider._get_mock_data("AAPL", "30d")
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 30
        assert all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])


class TestDataProviderFactory:
    """Test the data provider factory function"""
    
    def test_get_mock_provider_when_testing(self):
        """Test that mock provider is returned when TESTING env var is set"""
        os.environ['TESTING'] = 'true'
        provider = get_data_provider()
        assert isinstance(provider, MockProvider)
        del os.environ['TESTING']
    
    def test_get_yfinance_provider_by_default(self):
        """Test that YFinance provider is returned by default"""
        # Clear any environment variables
        os.environ.pop('TESTING', None)
        os.environ.pop('ALPACA_API_KEY', None)
        
        # Reset global instance
        import bot_v2.data_providers
        bot_v2.data_providers._provider_instance = None
        
        provider = get_data_provider()
        assert isinstance(provider, YFinanceProvider)
    
    def test_explicit_provider_type(self):
        """Test requesting specific provider type"""
        mock_provider = get_data_provider('mock')
        assert isinstance(mock_provider, MockProvider)
        
        yf_provider = get_data_provider('yfinance')
        assert isinstance(yf_provider, YFinanceProvider)
    
    def test_set_data_provider(self):
        """Test setting custom provider"""
        custom_provider = MockProvider()
        set_data_provider(custom_provider)
        
        provider = get_data_provider()
        assert provider is custom_provider


class TestDataProviderInterface:
    """Test that all providers implement the interface correctly"""
    
    @pytest.mark.parametrize("provider_class", [MockProvider, YFinanceProvider])
    def test_provider_has_required_methods(self, provider_class):
        """Test that provider has all required methods"""
        provider = provider_class()
        
        assert hasattr(provider, 'get_historical_data')
        assert hasattr(provider, 'get_current_price')
        assert hasattr(provider, 'get_multiple_symbols')
        assert hasattr(provider, 'is_market_open')
        
        # All methods should be callable
        assert callable(provider.get_historical_data)
        assert callable(provider.get_current_price)
        assert callable(provider.get_multiple_symbols)
        assert callable(provider.is_market_open)


class TestDataQuality:
    """Test data quality and consistency"""
    
    def test_ohlc_relationships(self):
        """Test that OHLC data maintains proper relationships"""
        provider = MockProvider()
        data = provider.get_historical_data("AAPL", "30d")
        
        # High should be >= Low
        assert (data['High'] >= data['Low']).all()
        
        # High should be >= Open and Close
        assert (data['High'] >= data['Open']).all()
        assert (data['High'] >= data['Close']).all()
        
        # Low should be <= Open and Close
        assert (data['Low'] <= data['Open']).all()
        assert (data['Low'] <= data['Close']).all()
        
        # Volume should be positive
        assert (data['Volume'] > 0).all()
    
    def test_date_ordering(self):
        """Test that dates are properly ordered"""
        provider = MockProvider()
        data = provider.get_historical_data("AAPL", "30d")
        
        # Dates should be in ascending order
        dates = data.index.to_list()
        assert dates == sorted(dates)
    
    def test_no_missing_data(self):
        """Test that there are no missing values"""
        provider = MockProvider()
        data = provider.get_historical_data("AAPL", "30d")
        
        assert not data.isnull().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])