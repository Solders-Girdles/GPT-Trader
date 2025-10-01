"""
Comprehensive tests for paper trading data feed.

Tests cover:
- DataFeed initialization
- Historical data loading
- Latest price fetching
- Data caching and updates
- Market hours detection
- Symbol management (add/remove)
- Error handling
- Edge cases
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from bot_v2.features.paper_trade.data import DataFeed


# ============================================================================
# Helper Functions
# ============================================================================


def create_sample_data(num_rows=30, base_price=100.0):
    """Create sample OHLCV data for testing."""
    import numpy as np

    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=num_rows, freq="D")
    prices = base_price + np.cumsum(np.random.randn(num_rows) * 2)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.02,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, num_rows),
        },
        index=dates,
    )


def create_mock_provider(historical_data=None, current_price=None):
    """Create mock data provider."""
    provider = Mock()

    if historical_data is not None:
        provider.get_historical_data = Mock(return_value=historical_data)
    else:
        provider.get_historical_data = Mock(return_value=create_sample_data())

    provider.get_current_price = Mock(return_value=current_price)

    return provider


# ============================================================================
# Test: DataFeed Initialization
# ============================================================================


class TestDataFeedInitialization:
    """Test DataFeed initialization."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_initialization_single_symbol(self, mock_get_provider):
        """Test initialization with single symbol."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        assert feed.symbols == ["AAPL"]
        assert feed.lookback_days == 30
        assert feed.update_interval == 60
        assert isinstance(feed.data_cache, dict)
        assert isinstance(feed.last_update, dict)

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_initialization_multiple_symbols(self, mock_get_provider):
        """Test initialization with multiple symbols."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        symbols = ["AAPL", "MSFT", "GOOGL"]
        feed = DataFeed(symbols=symbols)

        assert feed.symbols == symbols
        # Should fetch data for all symbols
        assert mock_provider.get_historical_data.call_count == len(symbols)

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_initialization_custom_lookback(self, mock_get_provider):
        """Test initialization with custom lookback period."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"], lookback_days=60)

        assert feed.lookback_days == 60

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_initialization_loads_historical_data(self, mock_get_provider):
        """Test that initialization loads historical data."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        assert "AAPL" in feed.data_cache
        assert not feed.data_cache["AAPL"].empty
        assert "AAPL" in feed.last_update

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_initialization_handles_empty_data(self, mock_get_provider):
        """Test initialization handles empty data gracefully."""
        mock_provider = create_mock_provider(historical_data=pd.DataFrame())
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["INVALID"])

        # Should create empty cache entry
        assert "INVALID" in feed.data_cache
        assert feed.data_cache["INVALID"].empty

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_initialization_handles_provider_error(self, mock_get_provider):
        """Test initialization handles data provider errors."""
        mock_provider = Mock()
        mock_provider.get_historical_data.side_effect = Exception("API error")
        mock_get_provider.return_value = mock_provider

        # Should not raise exception
        feed = DataFeed(symbols=["AAPL"])

        assert "AAPL" in feed.data_cache
        assert feed.data_cache["AAPL"].empty


# ============================================================================
# Test: Latest Price Fetching
# ============================================================================


class TestLatestPriceFetching:
    """Test latest price fetching."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_latest_price_from_cache(self, mock_get_provider):
        """Test getting latest price from cached data."""
        data = create_sample_data(base_price=150.0)
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Mock market hours to return False (use cache)
        with patch.object(feed, "_is_market_hours", return_value=False):
            price = feed.get_latest_price("AAPL")

        assert price is not None
        assert isinstance(price, float)

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_latest_price_during_market_hours(self, mock_get_provider):
        """Test getting latest price during market hours."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data, current_price=155.50)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Mock market hours to return True
        with patch.object(feed, "_is_market_hours", return_value=True):
            price = feed.get_latest_price("AAPL")

        assert price == 155.50
        mock_provider.get_current_price.assert_called_once_with("AAPL")

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_latest_price_market_hours_fallback(self, mock_get_provider):
        """Test fallback to cache when real-time fetch fails."""
        data = create_sample_data(base_price=150.0)
        mock_provider = create_mock_provider(historical_data=data, current_price=None)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        with patch.object(feed, "_is_market_hours", return_value=True):
            price = feed.get_latest_price("AAPL")

        # Should fall back to cached price
        assert price is not None
        assert isinstance(price, float)

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_latest_price_unknown_symbol(self, mock_get_provider):
        """Test getting latest price for unknown symbol."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        price = feed.get_latest_price("UNKNOWN")

        assert price is None

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_latest_price_empty_cache(self, mock_get_provider):
        """Test getting latest price when cache is empty."""
        mock_provider = create_mock_provider(historical_data=pd.DataFrame())
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        price = feed.get_latest_price("AAPL")

        assert price is None

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_latest_price_realtime_exception(self, mock_get_provider):
        """Test handling exception during real-time price fetch."""
        data = create_sample_data(base_price=150.0)
        mock_provider = create_mock_provider(historical_data=data)
        mock_provider.get_current_price.side_effect = Exception("API error")
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        with patch.object(feed, "_is_market_hours", return_value=True):
            price = feed.get_latest_price("AAPL")

        # Should fall back to cached price
        assert price is not None


# ============================================================================
# Test: Historical Data Retrieval
# ============================================================================


class TestHistoricalDataRetrieval:
    """Test historical data retrieval."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_historical_all_data(self, mock_get_provider):
        """Test getting all historical data."""
        data = create_sample_data(num_rows=50)
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        historical = feed.get_historical("AAPL")

        assert len(historical) == 50
        assert "close" in historical.columns

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_historical_with_periods(self, mock_get_provider):
        """Test getting specific number of periods."""
        data = create_sample_data(num_rows=50)
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        historical = feed.get_historical("AAPL", periods=10)

        assert len(historical) == 10

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_historical_periods_exceeds_available(self, mock_get_provider):
        """Test requesting more periods than available."""
        data = create_sample_data(num_rows=20)
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        historical = feed.get_historical("AAPL", periods=50)

        # Should return all available data
        assert len(historical) == 20

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_historical_unknown_symbol(self, mock_get_provider):
        """Test getting historical data for unknown symbol."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        historical = feed.get_historical("UNKNOWN")

        assert historical.empty

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_historical_returns_copy(self, mock_get_provider):
        """Test that get_historical returns a copy."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        historical1 = feed.get_historical("AAPL")
        historical2 = feed.get_historical("AAPL")

        # Should be independent copies
        assert historical1 is not historical2


# ============================================================================
# Test: Data Updates
# ============================================================================


class TestDataUpdates:
    """Test data update functionality."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_update_single_symbol(self, mock_get_provider):
        """Test updating single symbol."""
        initial_data = create_sample_data(num_rows=30)
        new_data = create_sample_data(num_rows=1, base_price=160.0)

        mock_provider = Mock()
        mock_provider.get_historical_data.side_effect = [initial_data, new_data]
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Force update by clearing last_update timestamp
        feed.last_update["AAPL"] = datetime.now() - timedelta(seconds=120)

        feed.update("AAPL")

        # Should have called provider twice (init + update)
        assert mock_provider.get_historical_data.call_count == 2

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_update_all_symbols(self, mock_get_provider):
        """Test updating all symbols."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL", "MSFT"])

        # Force updates
        for symbol in feed.symbols:
            feed.last_update[symbol] = datetime.now() - timedelta(seconds=120)

        feed.update()

        # Should update both symbols (init + update for each)
        assert mock_provider.get_historical_data.call_count >= 2

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_update_respects_interval(self, mock_get_provider):
        """Test that update respects update interval."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        initial_call_count = mock_provider.get_historical_data.call_count

        # Try to update immediately (should skip)
        feed.update("AAPL")

        # Should not make additional call
        assert mock_provider.get_historical_data.call_count == initial_call_count

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_update_maintains_lookback_window(self, mock_get_provider):
        """Test that update maintains lookback window."""
        initial_data = create_sample_data(num_rows=30)
        new_data = create_sample_data(num_rows=1, base_price=160.0)

        mock_provider = Mock()
        mock_provider.get_historical_data.side_effect = [initial_data, new_data]
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"], lookback_days=30)

        # Force update
        feed.last_update["AAPL"] = datetime.now() - timedelta(seconds=120)
        feed.update("AAPL")

        # Should maintain lookback window
        assert len(feed.data_cache["AAPL"]) <= 30

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_update_handles_error(self, mock_get_provider):
        """Test that update handles errors gracefully."""
        data = create_sample_data()
        mock_provider = Mock()
        mock_provider.get_historical_data.side_effect = [
            data,
            Exception("API error"),
        ]
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Force update
        feed.last_update["AAPL"] = datetime.now() - timedelta(seconds=120)

        # Should not raise exception
        feed.update("AAPL")

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_update_with_empty_new_data(self, mock_get_provider):
        """Test update when new data is empty."""
        initial_data = create_sample_data(num_rows=30)
        empty_data = pd.DataFrame()

        mock_provider = Mock()
        mock_provider.get_historical_data.side_effect = [initial_data, empty_data]
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        original_len = len(feed.data_cache["AAPL"])

        # Force update
        feed.last_update["AAPL"] = datetime.now() - timedelta(seconds=120)
        feed.update("AAPL")

        # Cache should remain unchanged
        assert len(feed.data_cache["AAPL"]) == original_len


# ============================================================================
# Test: Market Hours Detection
# ============================================================================


class TestMarketHoursDetection:
    """Test market hours detection."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_is_market_hours_weekday_morning(self, mock_get_provider):
        """Test market hours detection on weekday morning."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Monday at 10:00 AM
        with patch("bot_v2.features.paper_trade.data.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 8, 10, 0)  # Monday
            result = feed._is_market_hours()

        assert result is True

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_is_market_hours_weekday_afternoon(self, mock_get_provider):
        """Test market hours detection on weekday afternoon."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Monday at 2:00 PM
        with patch("bot_v2.features.paper_trade.data.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 8, 14, 0)  # Monday
            result = feed._is_market_hours()

        assert result is True

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_is_market_hours_before_open(self, mock_get_provider):
        """Test market hours detection before market opens."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Monday at 8:00 AM (before open)
        with patch("bot_v2.features.paper_trade.data.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 8, 8, 0)  # Monday
            result = feed._is_market_hours()

        assert result is False

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_is_market_hours_after_close(self, mock_get_provider):
        """Test market hours detection after market closes."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Monday at 5:00 PM (after close)
        with patch("bot_v2.features.paper_trade.data.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 8, 17, 0)  # Monday
            result = feed._is_market_hours()

        assert result is False

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_is_market_hours_saturday(self, mock_get_provider):
        """Test market hours detection on Saturday."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Saturday at 10:00 AM
        with patch("bot_v2.features.paper_trade.data.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 13, 10, 0)  # Saturday
            result = feed._is_market_hours()

        assert result is False

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_is_market_hours_sunday(self, mock_get_provider):
        """Test market hours detection on Sunday."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Sunday at 10:00 AM
        with patch("bot_v2.features.paper_trade.data.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 14, 10, 0)  # Sunday
            result = feed._is_market_hours()

        assert result is False


# ============================================================================
# Test: Symbol Management
# ============================================================================


class TestSymbolManagement:
    """Test adding and removing symbols."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_add_symbol(self, mock_get_provider):
        """Test adding a new symbol."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        feed.add_symbol("MSFT")

        assert "MSFT" in feed.symbols
        assert "MSFT" in feed.data_cache

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_add_existing_symbol(self, mock_get_provider):
        """Test adding a symbol that already exists."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        original_count = len(feed.symbols)
        feed.add_symbol("AAPL")

        # Should not add duplicate
        assert len(feed.symbols) == original_count

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_remove_symbol(self, mock_get_provider):
        """Test removing a symbol."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL", "MSFT"])

        feed.remove_symbol("MSFT")

        assert "MSFT" not in feed.symbols
        assert "MSFT" not in feed.data_cache
        assert "MSFT" not in feed.last_update

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_remove_nonexistent_symbol(self, mock_get_provider):
        """Test removing a symbol that doesn't exist."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Should not raise error
        feed.remove_symbol("NONEXISTENT")


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_initialization_empty_symbols_list(self, mock_get_provider):
        """Test initialization with empty symbols list."""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=[])

        assert feed.symbols == []
        assert len(feed.data_cache) == 0

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_historical_zero_periods(self, mock_get_provider):
        """Test getting historical data with zero periods."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        historical = feed.get_historical("AAPL", periods=0)

        # Should return empty or full data based on implementation
        assert isinstance(historical, pd.DataFrame)

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_get_historical_negative_periods(self, mock_get_provider):
        """Test getting historical data with negative periods."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        historical = feed.get_historical("AAPL", periods=-10)

        # Should handle gracefully
        assert isinstance(historical, pd.DataFrame)

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_very_large_lookback_period(self, mock_get_provider):
        """Test with very large lookback period."""
        data = create_sample_data(num_rows=100)
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"], lookback_days=1000)

        assert feed.lookback_days == 1000

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_zero_lookback_period(self, mock_get_provider):
        """Test with zero lookback period."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"], lookback_days=0)

        assert feed.lookback_days == 0

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_column_name_standardization(self, mock_get_provider):
        """Test that column names are standardized to lowercase."""
        # Data with uppercase columns
        data = pd.DataFrame(
            {
                "OPEN": [100, 101],
                "HIGH": [102, 103],
                "LOW": [99, 100],
                "CLOSE": [101, 102],
                "VOLUME": [1000000, 1100000],
            }
        )

        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Should have lowercase columns
        assert "close" in feed.data_cache["AAPL"].columns
        assert "CLOSE" not in feed.data_cache["AAPL"].columns


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_complete_workflow(self, mock_get_provider):
        """Test complete data feed workflow."""
        initial_data = create_sample_data(num_rows=30, base_price=150.0)
        update_data = create_sample_data(num_rows=1, base_price=155.0)

        mock_provider = Mock()
        mock_provider.get_historical_data.side_effect = [
            initial_data,
            initial_data,
            update_data,
            update_data,
        ]
        mock_provider.get_current_price.return_value = 156.0
        mock_get_provider.return_value = mock_provider

        # Initialize with multiple symbols
        feed = DataFeed(symbols=["AAPL", "MSFT"])

        # Get latest prices
        with patch.object(feed, "_is_market_hours", return_value=False):
            price1 = feed.get_latest_price("AAPL")
            price2 = feed.get_latest_price("MSFT")

        assert price1 is not None
        assert price2 is not None

        # Get historical data
        hist = feed.get_historical("AAPL", periods=10)
        assert len(hist) == 10

        # Update data
        feed.last_update["AAPL"] = datetime.now() - timedelta(seconds=120)
        feed.update("AAPL")

    @patch("bot_v2.features.paper_trade.data.get_data_provider")
    def test_multiple_symbol_management(self, mock_get_provider):
        """Test managing multiple symbols dynamically."""
        data = create_sample_data()
        mock_provider = create_mock_provider(historical_data=data)
        mock_get_provider.return_value = mock_provider

        feed = DataFeed(symbols=["AAPL"])

        # Add symbols
        feed.add_symbol("MSFT")
        feed.add_symbol("GOOGL")

        assert len(feed.symbols) == 3

        # Remove symbol
        feed.remove_symbol("MSFT")

        assert len(feed.symbols) == 2
        assert "MSFT" not in feed.symbols
