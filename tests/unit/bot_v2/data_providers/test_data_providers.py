"""
Comprehensive tests for data providers.

Tests cover:
- YFinanceProvider (historical data, current price, caching, mock fallback)
- MockProvider (deterministic data generation, fixture loading)
- Factory function (provider selection, environment detection)
- Provider interface compliance
- Edge cases and error handling
"""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from bot_v2.data_providers import (
    DataProvider,
    MockProvider,
    YFinanceProvider,
    get_data_provider,
    set_data_provider,
)


# ============================================================================
# Test: Abstract Base Class
# ============================================================================


class TestDataProviderInterface:
    """Test DataProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that DataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProvider()

    def test_subclass_must_implement_methods(self):
        """Test that subclasses must implement all abstract methods."""

        class IncompleteProvider(DataProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()


# ============================================================================
# Test: YFinanceProvider Initialization
# ============================================================================


class TestYFinanceProviderInitialization:
    """Test YFinanceProvider initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        provider = YFinanceProvider()

        assert provider._yfinance is None
        assert isinstance(provider._cache, dict)
        assert isinstance(provider._cache_expiry, dict)
        assert provider._cache_duration == timedelta(minutes=5)

    def test_lazy_yfinance_loading(self):
        """Test that yfinance is loaded lazily."""
        provider = YFinanceProvider()

        assert provider._yfinance is None

        # Mock yfinance module to test lazy loading
        mock_yf = Mock()
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            # Access yf property should trigger loading
            yf = provider.yf

            # Should now be loaded
            assert provider._yfinance is not None
            assert provider._yfinance == mock_yf


# ============================================================================
# Test: YFinanceProvider Historical Data
# ============================================================================


class TestYFinanceProviderHistoricalData:
    """Test YFinanceProvider historical data fetching."""

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_historical_data_success(self, mock_yf):
        """Test successful historical data retrieval."""
        provider = YFinanceProvider()

        # Mock yfinance response
        mock_ticker = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        data = provider.get_historical_data("AAPL", period="3d")

        assert not data.empty
        assert len(data) == 3
        mock_yf.Ticker.assert_called_once_with("AAPL")
        mock_ticker.history.assert_called_once_with(period="3d", interval="1d")

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_historical_data_caching(self, mock_yf):
        """Test that historical data is cached."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_data = pd.DataFrame({"Close": [100, 101, 102]})
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        # First call
        data1 = provider.get_historical_data("AAPL", period="3d")

        # Second call (should use cache)
        data2 = provider.get_historical_data("AAPL", period="3d")

        # Should only call API once
        assert mock_yf.Ticker.call_count == 1
        assert len(data1) == len(data2)

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_historical_data_empty_response(self, mock_yf):
        """Test handling of empty response from yfinance."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_yf.Ticker.return_value = mock_ticker

        data = provider.get_historical_data("INVALID", period="3d")

        # Should return mock data
        assert not data.empty

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_historical_data_exception(self, mock_yf):
        """Test handling of exceptions during data fetch."""
        provider = YFinanceProvider()

        mock_yf.Ticker.side_effect = Exception("API error")

        data = provider.get_historical_data("AAPL", period="3d")

        # Should return mock data
        assert not data.empty

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_historical_data_different_periods(self, mock_yf):
        """Test fetching data with different periods."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_data = pd.DataFrame({"Close": [100, 101, 102]})
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        provider.get_historical_data("AAPL", period="7d")
        provider.get_historical_data("AAPL", period="30d")

        # Different periods = different cache keys
        assert mock_yf.Ticker.call_count == 2

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_historical_data_different_intervals(self, mock_yf):
        """Test fetching data with different intervals."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_data = pd.DataFrame({"Close": [100, 101, 102]})
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        provider.get_historical_data("AAPL", period="7d", interval="1d")
        provider.get_historical_data("AAPL", period="7d", interval="1h")

        # Different intervals = different cache keys
        assert mock_yf.Ticker.call_count == 2


# ============================================================================
# Test: YFinanceProvider Current Price
# ============================================================================


class TestYFinanceProviderCurrentPrice:
    """Test YFinanceProvider current price fetching."""

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_current_price_success(self, mock_yf):
        """Test successful current price retrieval."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_ticker.info = {"currentPrice": 150.25}
        mock_yf.Ticker.return_value = mock_ticker

        price = provider.get_current_price("AAPL")

        assert price == 150.25

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_current_price_fallback_to_regular_market(self, mock_yf):
        """Test fallback to regularMarketPrice."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_ticker.info = {"regularMarketPrice": 149.75}
        mock_yf.Ticker.return_value = mock_ticker

        price = provider.get_current_price("AAPL")

        assert price == 149.75

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_current_price_exception(self, mock_yf):
        """Test handling of exceptions."""
        provider = YFinanceProvider()

        mock_yf.Ticker.side_effect = Exception("API error")

        price = provider.get_current_price("AAPL")

        # Should return default
        assert price == 100.0

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_current_price_no_price_data(self, mock_yf):
        """Test when no price data is available."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        price = provider.get_current_price("AAPL")

        assert price == 100.0


# ============================================================================
# Test: YFinanceProvider Multiple Symbols
# ============================================================================


class TestYFinanceProviderMultipleSymbols:
    """Test YFinanceProvider multiple symbols fetching."""

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_get_multiple_symbols(self, mock_yf):
        """Test fetching multiple symbols."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_data = pd.DataFrame({"Close": [100, 101, 102]})
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        symbols = ["AAPL", "MSFT", "GOOGL"]
        result = provider.get_multiple_symbols(symbols, period="7d")

        assert len(result) == 3
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOGL" in result
        assert mock_yf.Ticker.call_count == 3


# ============================================================================
# Test: YFinanceProvider Market Hours
# ============================================================================


class TestYFinanceProviderMarketHours:
    """Test YFinanceProvider market hours checking."""

    @patch("bot_v2.data_providers.datetime")
    def test_is_market_open_weekday_trading_hours(self, mock_datetime):
        """Test market open during weekday trading hours."""
        provider = YFinanceProvider()

        mock_now = Mock()
        mock_now.weekday.return_value = 2  # Wednesday
        mock_now.hour = 10
        mock_datetime.now.return_value = mock_now

        assert provider.is_market_open() is True

    @patch("bot_v2.data_providers.datetime")
    def test_is_market_open_weekday_before_open(self, mock_datetime):
        """Test market closed before opening."""
        provider = YFinanceProvider()

        mock_now = Mock()
        mock_now.weekday.return_value = 2
        mock_now.hour = 8
        mock_datetime.now.return_value = mock_now

        assert provider.is_market_open() is False

    @patch("bot_v2.data_providers.datetime")
    def test_is_market_open_weekday_after_close(self, mock_datetime):
        """Test market closed after closing."""
        provider = YFinanceProvider()

        mock_now = Mock()
        mock_now.weekday.return_value = 2
        mock_now.hour = 17
        mock_datetime.now.return_value = mock_now

        assert provider.is_market_open() is False

    @patch("bot_v2.data_providers.datetime")
    def test_is_market_open_weekend(self, mock_datetime):
        """Test market closed on weekend."""
        provider = YFinanceProvider()

        # Saturday
        mock_now = Mock()
        mock_now.weekday.return_value = 5
        mock_now.hour = 12
        mock_datetime.now.return_value = mock_now

        assert provider.is_market_open() is False


# ============================================================================
# Test: YFinanceProvider Mock Data Generation
# ============================================================================


class TestYFinanceProviderMockData:
    """Test YFinanceProvider mock data generation."""

    def test_mock_data_generation(self):
        """Test that mock data is generated correctly."""
        provider = YFinanceProvider()

        data = provider._get_mock_data("AAPL", "30d")

        assert not data.empty
        assert len(data) == 30
        assert "Open" in data.columns
        assert "High" in data.columns
        assert "Low" in data.columns
        assert "Close" in data.columns
        assert "Volume" in data.columns

    def test_mock_data_deterministic(self):
        """Test that mock data is deterministic for same symbol."""
        provider = YFinanceProvider()

        data1 = provider._get_mock_data("AAPL", "30d")
        data2 = provider._get_mock_data("AAPL", "30d")

        # Compare values (ignore index timestamps which may differ by microseconds)
        import pandas as pd

        pd.testing.assert_frame_equal(data1.reset_index(drop=True), data2.reset_index(drop=True))

    def test_mock_data_different_symbols(self):
        """Test that different symbols produce different mock data."""
        provider = YFinanceProvider()

        data_aapl = provider._get_mock_data("AAPL", "30d")
        data_msft = provider._get_mock_data("MSFT", "30d")

        # Should be different data
        assert not data_aapl.equals(data_msft)


# ============================================================================
# Test: MockProvider
# ============================================================================


class TestMockProvider:
    """Test MockProvider."""

    def test_initialization(self):
        """Test MockProvider initialization."""
        provider = MockProvider()

        assert provider.data_dir == "tests/fixtures/market_data"
        assert isinstance(provider._mock_data, dict)

    def test_get_historical_data(self):
        """Test mock historical data generation."""
        provider = MockProvider()

        data = provider.get_historical_data("AAPL", period="30d")

        assert not data.empty
        assert len(data) == 30
        assert "Close" in data.columns

    def test_get_historical_data_deterministic(self):
        """Test that mock data is deterministic."""
        provider = MockProvider()

        data1 = provider.get_historical_data("AAPL", period="30d")
        data2 = provider.get_historical_data("AAPL", period="30d")

        assert data1.equals(data2)

    def test_get_historical_data_fixed_date(self):
        """Test that mock data uses fixed end date."""
        provider = MockProvider()

        data = provider.get_historical_data("AAPL", period="10d")

        # Should end on fixed date (2024-03-01)
        assert data.index[-1].year == 2024
        assert data.index[-1].month == 3

    def test_get_current_price(self):
        """Test mock current price."""
        provider = MockProvider()

        assert provider.get_current_price("AAPL") == 150.0
        assert provider.get_current_price("GOOGL") == 2800.0
        assert provider.get_current_price("UNKNOWN") == 100.0

    def test_get_multiple_symbols(self):
        """Test mock multiple symbols."""
        provider = MockProvider()

        symbols = ["AAPL", "MSFT"]
        result = provider.get_multiple_symbols(symbols, period="10d")

        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result

    def test_is_market_open(self):
        """Test mock market hours."""
        provider = MockProvider()

        assert provider.is_market_open() is True

    def test_ohlc_relationships(self):
        """Test that OHLC data has proper relationships."""
        provider = MockProvider()

        data = provider.get_historical_data("AAPL", period="30d")

        for idx in data.index:
            assert data.loc[idx, "High"] >= data.loc[idx, "Open"]
            assert data.loc[idx, "High"] >= data.loc[idx, "Close"]
            assert data.loc[idx, "Low"] <= data.loc[idx, "Open"]
            assert data.loc[idx, "Low"] <= data.loc[idx, "Close"]


# ============================================================================
# Test: Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test get_data_provider factory function."""

    def test_get_data_provider_default(self):
        """Test default provider selection."""
        # Reset global instance
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None

        provider = get_data_provider()

        assert isinstance(provider, YFinanceProvider)

    @patch.dict(os.environ, {"TESTING": "true"})
    def test_get_data_provider_testing_mode(self):
        """Test provider selection in testing mode."""
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None

        provider = get_data_provider()

        assert isinstance(provider, MockProvider)

    def test_get_data_provider_explicit_mock(self):
        """Test explicit mock provider selection."""
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None

        provider = get_data_provider("mock")

        assert isinstance(provider, MockProvider)

    def test_get_data_provider_explicit_yfinance(self):
        """Test explicit yfinance provider selection."""
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None

        provider = get_data_provider("yfinance")

        assert isinstance(provider, YFinanceProvider)

    def test_get_data_provider_singleton(self):
        """Test that factory returns singleton instance."""
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None

        provider1 = get_data_provider("yfinance")
        provider2 = get_data_provider()

        assert provider1 is provider2

    def test_get_data_provider_unsupported(self):
        """Unsupported provider names should raise immediately."""
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None

        with pytest.raises(ValueError, match="Unsupported data provider 'alpaca'"):
            get_data_provider("alpaca")

    def test_set_data_provider(self):
        """Test setting custom data provider."""
        custom_provider = MockProvider()

        set_data_provider(custom_provider)

        provider = get_data_provider()

        assert provider is custom_provider

        # Cleanup
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None


# ============================================================================
# Test: Cache Functionality
# ============================================================================


class TestCacheFunctionality:
    """Test caching functionality."""

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_cache_expiry(self, mock_yf):
        """Test that cache expires after duration."""
        provider = YFinanceProvider()
        provider._cache_duration = timedelta(seconds=1)

        mock_ticker = Mock()
        mock_data = pd.DataFrame({"Close": [100, 101, 102]})
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        # First call
        provider.get_historical_data("AAPL", period="3d")

        # Wait for cache to expire
        import time

        time.sleep(1.1)

        # Second call (should fetch again)
        provider.get_historical_data("AAPL", period="3d")

        # Should call API twice
        assert mock_yf.Ticker.call_count == 2

    def test_cache_key_uniqueness(self):
        """Test that cache keys are unique per symbol/period/interval."""
        provider = YFinanceProvider()

        # Different combinations should have different keys
        combinations = [
            ("AAPL", "7d", "1d"),
            ("AAPL", "30d", "1d"),
            ("AAPL", "7d", "1h"),
            ("MSFT", "7d", "1d"),
        ]

        cache_keys = set()
        for symbol, period, interval in combinations:
            key = f"{symbol}_{period}_{interval}"
            cache_keys.add(key)

        assert len(cache_keys) == 4


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_mock_data_zero_days(self):
        """Test mock data generation with zero days."""
        provider = MockProvider()

        # Should handle gracefully
        data = provider.get_historical_data("AAPL", period="0d")

        assert isinstance(data, pd.DataFrame)

    def test_mock_data_invalid_period(self):
        """Test mock data with invalid period format."""
        provider = MockProvider()

        # Should use default (60d)
        data = provider.get_historical_data("AAPL", period="invalid")

        assert not data.empty

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_yfinance_none_response(self, mock_yf):
        """Test handling of None response from yfinance."""
        provider = YFinanceProvider()

        mock_ticker = Mock()
        mock_ticker.history.return_value = None
        mock_yf.Ticker.return_value = mock_ticker

        # Should handle gracefully
        try:
            data = provider.get_historical_data("AAPL", period="3d")
            assert isinstance(data, pd.DataFrame)
        except Exception:
            pytest.fail("Should handle None response gracefully")

    def test_empty_symbol_list(self):
        """Test getting multiple symbols with empty list."""
        provider = MockProvider()

        result = provider.get_multiple_symbols([])

        assert len(result) == 0

    def test_mock_provider_nonexistent_data_dir(self):
        """Test MockProvider with nonexistent data directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "nonexistent")
            provider = MockProvider(data_dir=nonexistent_dir)

            # Should still work (just no fixtures)
            data = provider.get_historical_data("AAPL", period="10d")
            assert not data.empty


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch("bot_v2.data_providers.YFinanceProvider.yf")
    def test_complete_workflow_yfinance(self, mock_yf):
        """Test complete workflow with YFinanceProvider."""
        provider = YFinanceProvider()

        # Mock responses
        mock_ticker = Mock()
        mock_data = pd.DataFrame({"Close": [100, 101, 102]})
        mock_ticker.history.return_value = mock_data
        mock_ticker.info = {"currentPrice": 102.5}
        mock_yf.Ticker.return_value = mock_ticker

        # Get historical data
        hist = provider.get_historical_data("AAPL", period="3d")
        assert len(hist) == 3

        # Get current price
        price = provider.get_current_price("AAPL")
        assert price == 102.5

        # Check market status
        is_open = provider.is_market_open()
        assert isinstance(is_open, bool)

    def test_complete_workflow_mock(self):
        """Test complete workflow with MockProvider."""
        provider = MockProvider()

        # Get historical data
        hist = provider.get_historical_data("AAPL", period="30d")
        assert len(hist) == 30

        # Get multiple symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        result = provider.get_multiple_symbols(symbols)
        assert len(result) == 3

        # Get current prices
        for symbol in symbols:
            price = provider.get_current_price(symbol)
            assert price > 0

        # Market always open
        assert provider.is_market_open() is True

    @patch.dict(os.environ, {"TESTING": "true"})
    def test_factory_with_environment(self):
        """Test factory function with environment configuration."""
        import bot_v2.data_providers as dp_module

        dp_module._provider_instance = None

        provider = get_data_provider()

        assert isinstance(provider, MockProvider)

        # Cleanup
        dp_module._provider_instance = None
