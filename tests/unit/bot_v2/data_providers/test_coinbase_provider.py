"""Tests for Coinbase data provider."""

import os
import pandas as pd
import pytest
from unittest.mock import MagicMock, Mock, patch

from bot_v2.data_providers.coinbase_provider import (
    CoinbaseDataProvider,
    create_coinbase_provider,
)
from bot_v2.features.brokerages.coinbase.client import CoinbaseClient
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.market_data_service import (
    CoinbaseTickerService,
    TickerCache,
)


class TestCoinbaseDataProvider:
    """Test CoinbaseDataProvider class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Mock environment variables
        self.original_env = {}
        env_vars = {
            "COINBASE_SANDBOX": "0",
            "COINBASE_API_MODE": "advanced",
            "COINBASE_ENABLE_STREAMING": "0",
            "COINBASE_ENABLE_DERIVATIVES": "0",
            "COINBASE_DEFAULT_QUOTE": "USD",
        }
        for key, value in env_vars.items():
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = value

        # Mock client and adapter
        self.mock_client = Mock(spec=CoinbaseClient)
        self.mock_client.base_url = "https://api.coinbase.com"
        self.mock_client.api_mode = "advanced"

        self.mock_adapter = Mock()
        self.mock_adapter.get_candles.return_value = []
        self.mock_adapter.get_quote.return_value = Mock(last=50000.0)

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        # Restore environment variables
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_provider_init_default(self) -> None:
        """Test provider initialization with default parameters."""
        provider = CoinbaseDataProvider()

        assert provider.client is not None
        assert provider.adapter is not None
        assert provider.enable_streaming is False
        assert provider.cache_ttl == 5
        assert provider.ticker_service is None
        assert provider.ticker_cache is None

    def test_provider_init_with_client_and_adapter(self) -> None:
        """Test provider initialization with provided client and adapter."""
        provider = CoinbaseDataProvider(
            client=self.mock_client, adapter=self.mock_adapter
        )

        assert provider.client == self.mock_client
        assert provider.adapter == self.mock_adapter

    def test_provider_init_with_streaming_enabled(self) -> None:
        """Test provider initialization with streaming enabled."""
        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseTickerService"):
            with patch("bot_v2.data_providers.coinbase_provider.TickerCache"):
                provider = CoinbaseDataProvider(enable_streaming=True)

                assert provider.enable_streaming is True

    def test_provider_init_streaming_from_env(self) -> None:
        """Test provider initialization reads streaming from environment."""
        os.environ["COINBASE_ENABLE_STREAMING"] = "1"

        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseTickerService"):
            with patch("bot_v2.data_providers.coinbase_provider.TickerCache"):
                provider = CoinbaseDataProvider()

                assert provider.enable_streaming is True

    def test_setup_streaming_success(self) -> None:
        """Test successful streaming setup."""
        provider = CoinbaseDataProvider()
        provider.enable_streaming = True

        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseTickerService") as mock_service:
            with patch("bot_v2.data_providers.coinbase_provider.TickerCache") as mock_cache:
                provider._setup_streaming()

                mock_cache.assert_called_once()
                mock_service.assert_called_once()

    def test_setup_streaming_failure(self) -> None:
        """Test streaming setup failure falls back to REST."""
        provider = CoinbaseDataProvider()
        provider.enable_streaming = True

        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseTickerService", side_effect=Exception("Connection failed")):
            provider._setup_streaming()

            assert provider.enable_streaming is False

    def test_normalize_symbol_coinbase_format(self) -> None:
        """Test symbol normalization for already-formatted symbols."""
        provider = CoinbaseDataProvider()

        result = provider._normalize_symbol("BTC-USD")
        assert result == "BTC-USD"

        result = provider._normalize_symbol("ETH-PERP")
        assert result == "ETH-PERP"

    def test_normalize_symbol_spot_mapping(self) -> None:
        """Test symbol normalization for spot trading."""
        provider = CoinbaseDataProvider()

        result = provider._normalize_symbol("BTC")
        assert result == "BTC-USD"

        result = provider._normalize_symbol("ETH")
        assert result == "ETH-USD"

        result = provider._normalize_symbol("SOL")
        assert result == "SOL-USD"

    def test_normalize_symbol_derivatives_enabled(self) -> None:
        """Test symbol normalization with derivatives enabled."""
        os.environ["COINBASE_ENABLE_DERIVATIVES"] = "1"
        provider = CoinbaseDataProvider()

        result = provider._normalize_symbol("BTC")
        assert result == "BTC-PERP"

        result = provider._normalize_symbol("ETH")
        assert result == "ETH-PERP"

    def test_normalize_symbol_custom_quote(self) -> None:
        """Test symbol normalization with custom quote currency."""
        os.environ["COINBASE_DEFAULT_QUOTE"] = "USDC"
        provider = CoinbaseDataProvider()

        result = provider._normalize_symbol("BTC")
        assert result == "BTC-USDC"

    def test_normalize_symbol_unknown_symbol(self) -> None:
        """Test symbol normalization for unknown symbols."""
        provider = CoinbaseDataProvider()

        result = provider._normalize_symbol("UNKNOWN")
        assert result == "UNKNOWN-USD"

    def test_normalize_symbol_invalid_quote(self) -> None:
        """Test symbol normalization with invalid quote defaults to USD."""
        os.environ["COINBASE_DEFAULT_QUOTE"] = "INVALID"
        provider = CoinbaseDataProvider()

        result = provider._normalize_symbol("BTC")
        assert result == "BTC-USD"

    def test_get_historical_data_cache_hit(self) -> None:
        """Test historical data retrieval with cache hit."""
        provider = CoinbaseDataProvider()
        cached_df = pd.DataFrame({"Close": [100.0, 101.0]})
        provider._historical_cache["BTC-USD_60d_1d"] = (cached_df, pd.Timestamp.now())

        result = provider.get_historical_data("BTC", "60d", "1d")

        assert result.equals(cached_df)
        provider.mock_adapter.get_candles.assert_not_called()

    def test_get_historical_data_cache_miss_success(self) -> None:
        """Test historical data retrieval with cache miss and successful API call."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter)

        # Mock candle data
        mock_candle = Mock()
        mock_candle.ts = pd.Timestamp("2024-01-01")
        mock_candle.open = "50000.0"
        mock_candle.high = "51000.0"
        mock_candle.low = "49000.0"
        mock_candle.close = "50500.0"
        mock_candle.volume = "1000000.0"

        self.mock_adapter.get_candles.return_value = [mock_candle]

        result = provider.get_historical_data("BTC", "60d", "1d")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["Close"] == 50500.0
        assert "BTC-USD_60d_1d" in provider._historical_cache

    def test_get_historical_data_api_failure_fallback(self) -> None:
        """Test historical data retrieval falls back to mock on API failure."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter)
        self.mock_adapter.get_candles.side_effect = Exception("API Error")

        result = provider.get_historical_data("BTC", "60d", "1d")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Mock data should be returned

    def test_get_historical_data_empty_response_fallback(self) -> None:
        """Test historical data retrieval falls back to mock on empty response."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter)
        self.mock_adapter.get_candles.return_value = []

        result = provider.get_historical_data("BTC", "60d", "1d")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Mock data should be returned

    def test_get_historical_data_different_intervals(self) -> None:
        """Test historical data retrieval with different intervals."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter)

        mock_candle = Mock()
        mock_candle.ts = pd.Timestamp("2024-01-01")
        mock_candle.open = "50000.0"
        mock_candle.high = "51000.0"
        mock_candle.low = "49000.0"
        mock_candle.close = "50500.0"
        mock_candle.volume = "1000000.0"

        self.mock_adapter.get_candles.return_value = [mock_candle]

        # Test different intervals
        for interval in ["1m", "5m", "15m", "30m", "1h", "2h", "6h", "1d"]:
            result = provider.get_historical_data("BTC", "60d", interval)
            assert isinstance(result, pd.DataFrame)

            # Check that granularity was mapped correctly
            self.mock_adapter.get_candles.assert_called()
            call_args = self.mock_adapter.get_candles.call_args
            assert call_args[0][1] in ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", 
                                      "THIRTY_MINUTE", "ONE_HOUR", "TWO_HOUR", "SIX_HOUR", "ONE_DAY"]

    def test_get_current_price_websocket_fresh(self) -> None:
        """Test current price retrieval from fresh WebSocket cache."""
        provider = CoinbaseDataProvider(enable_streaming=True)
        
        # Mock ticker cache
        mock_ticker = Mock()
        mock_ticker.last = 50500.0
        mock_cache = Mock()
        mock_cache.get.return_value = mock_ticker
        mock_cache.is_stale.return_value = False
        provider.ticker_cache = mock_cache

        result = provider.get_current_price("BTC")

        assert result == 50500.0
        mock_cache.get.assert_called_with("BTC-USD")

    def test_get_current_price_websocket_stale_fallback(self) -> None:
        """Test current price retrieval falls back to REST when WebSocket data is stale."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter, enable_streaming=True)
        
        # Mock stale ticker cache
        mock_cache = Mock()
        mock_cache.get.return_value = Mock(last=50500.0)
        mock_cache.is_stale.return_value = True
        provider.ticker_cache = mock_cache

        result = provider.get_current_price("BTC")

        assert result == 50000.0  # From mock adapter
        self.mock_adapter.get_quote.assert_called_with("BTC-USD")

    def test_get_current_price_rest_fallback(self) -> None:
        """Test current price retrieval from REST API."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter)

        result = provider.get_current_price("BTC")

        assert result == 50000.0
        self.mock_adapter.get_quote.assert_called_with("BTC-USD")

    def test_get_current_price_api_failure_default(self) -> None:
        """Test current price retrieval returns default on API failure."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter)
        self.mock_adapter.get_quote.side_effect = Exception("API Error")

        result = provider.get_current_price("BTC")

        assert result == 100.0  # Default fallback value

    def test_get_multiple_symbols(self) -> None:
        """Test getting historical data for multiple symbols."""
        provider = CoinbaseDataProvider(adapter=self.mock_adapter)

        mock_candle = Mock()
        mock_candle.ts = pd.Timestamp("2024-01-01")
        mock_candle.open = "50000.0"
        mock_candle.high = "51000.0"
        mock_candle.low = "49000.0"
        mock_candle.close = "50500.0"
        mock_candle.volume = "1000000.0"

        self.mock_adapter.get_candles.return_value = [mock_candle]

        symbols = ["BTC", "ETH", "SOL"]
        result = provider.get_multiple_symbols(symbols, "60d")

        assert len(result) == 3
        for symbol in symbols:
            assert symbol in result
            assert isinstance(result[symbol], pd.DataFrame)

    def test_get_multiple_symbols_with_streaming(self) -> None:
        """Test getting multiple symbols with streaming enabled."""
        provider = CoinbaseDataProvider(enable_streaming=True)
        
        # Mock ticker service
        mock_service = Mock()
        mock_service._thread = Mock()
        mock_service._thread.is_alive.return_value = False
        provider.ticker_service = mock_service

        with patch.object(provider, "get_historical_data") as mock_get_data:
            mock_get_data.return_value = pd.DataFrame()
            
            provider.get_multiple_symbols(["BTC", "ETH"], "60d")

            # Should update symbols in ticker service
            assert mock_service._symbols == ["BTC-USD", "ETH-USD"]

    def test_is_market_open(self) -> None:
        """Test market open check."""
        provider = CoinbaseDataProvider()

        result = provider.is_market_open()

        assert result is True  # Crypto markets are always open

    def test_get_mock_data_btc(self) -> None:
        """Test mock data generation for BTC."""
        provider = CoinbaseDataProvider()

        result = provider._get_mock_data("BTC", "60d")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 60
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        # BTC should have higher base price
        assert result["Close"].mean() > 10000

    def test_get_mock_data_eth(self) -> None:
        """Test mock data generation for ETH."""
        provider = CoinbaseDataProvider()

        result = provider._get_mock_data("ETH", "30d")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 30
        # ETH should have lower base price than BTC but higher than default
        assert 1000 < result["Close"].mean() < 10000

    def test_get_mock_data_other_symbol(self) -> None:
        """Test mock data generation for other symbols."""
        provider = CoinbaseDataProvider()

        result = provider._get_mock_data("UNKNOWN", "7d")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 7
        # Other symbols should have default base price
        assert result["Close"].mean() < 1000

    def test_start_stop_streaming(self) -> None:
        """Test starting and stopping streaming."""
        provider = CoinbaseDataProvider(enable_streaming=True)
        
        mock_service = Mock()
        provider.ticker_service = mock_service

        provider.start_streaming()
        mock_service.start.assert_called_once()

        provider.stop_streaming()
        mock_service.stop.assert_called_once()

    def test_context_manager(self) -> None:
        """Test context manager functionality."""
        provider = CoinbaseDataProvider(enable_streaming=True)
        
        mock_service = Mock()
        provider.ticker_service = mock_service

        with provider:
            mock_service.start.assert_called_once()

        mock_service.stop.assert_called_once()

    def test_on_ticker_update_callback(self) -> None:
        """Test ticker update callback."""
        provider = CoinbaseDataProvider()

        mock_ticker = Mock()
        mock_ticker.bid = 50000.0
        mock_ticker.ask = 50100.0
        mock_ticker.last = 50500.0

        # Should not raise any exceptions
        provider._on_ticker_update("BTC-USD", mock_ticker)


class TestCreateCoinbaseProvider:
    """Test factory function for creating Coinbase providers."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.original_env = {}
        for key in ["COINBASE_USE_REAL_DATA", "COINBASE_ENABLE_STREAMING"]:
            self.original_env[key] = os.environ.get(key)
            os.environ[key] = "0"

    def teardown_method(self) -> None:
        """Clean up test fixtures."""
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_create_mock_provider_env_disabled(self) -> None:
        """Test creating mock provider when real data is disabled."""
        os.environ["COINBASE_USE_REAL_DATA"] = "0"

        with patch("bot_v2.data_providers.coinbase_provider.MockProvider") as mock_provider:
            create_coinbase_provider()
            mock_provider.assert_called_once()

    def test_create_mock_provider_explicit_false(self) -> None:
        """Test creating mock provider when explicitly disabled."""
        with patch("bot_v2.data_providers.coinbase_provider.MockProvider") as mock_provider:
            create_coinbase_provider(use_real_data=False)
            mock_provider.assert_called_once()

    def test_create_real_provider_env_enabled(self) -> None:
        """Test creating real provider when enabled via environment."""
        os.environ["COINBASE_USE_REAL_DATA"] = "1"

        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseDataProvider") as mock_provider:
            create_coinbase_provider()
            mock_provider.assert_called_once_with(enable_streaming=False)

    def test_create_real_provider_explicit_true(self) -> None:
        """Test creating real provider when explicitly enabled."""
        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseDataProvider") as mock_provider:
            create_coinbase_provider(use_real_data=True)
            mock_provider.assert_called_once_with(enable_streaming=False)

    def test_create_real_provider_with_streaming(self) -> None:
        """Test creating real provider with streaming enabled."""
        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseDataProvider") as mock_provider:
            create_coinbase_provider(use_real_data=True, enable_streaming=True)
            mock_provider.assert_called_once_with(enable_streaming=True)

    def test_create_real_provider_streaming_from_env(self) -> None:
        """Test creating real provider reads streaming from environment."""
        os.environ["COINBASE_USE_REAL_DATA"] = "1"
        os.environ["COINBASE_ENABLE_STREAMING"] = "1"

        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseDataProvider") as mock_provider:
            create_coinbase_provider()
            mock_provider.assert_called_once_with(enable_streaming=True)


class TestIntegrationWorkflow:
    """Test integration workflows for Coinbase provider."""

    def test_full_workflow_historical_data(self) -> None:
        """Test full workflow for getting historical data."""
        mock_adapter = Mock()
        mock_candle = Mock()
        mock_candle.ts = pd.Timestamp("2024-01-01")
        mock_candle.open = "50000.0"
        mock_candle.high = "51000.0"
        mock_candle.low = "49000.0"
        mock_candle.close = "50500.0"
        mock_candle.volume = "1000000.0"
        mock_adapter.get_candles.return_value = [mock_candle]
        mock_adapter.get_quote.return_value = Mock(last=50500.0)

        provider = CoinbaseDataProvider(adapter=mock_adapter)

        # Get historical data
        hist_data = provider.get_historical_data("BTC", "30d", "1d")
        assert isinstance(hist_data, pd.DataFrame)
        assert len(hist_data) == 1

        # Get current price
        current_price = provider.get_current_price("BTC")
        assert current_price == 50500.0

        # Check cache is populated
        cache_key = "BTC-USD_30d_1d"
        assert cache_key in provider._historical_cache

    def test_full_workflow_with_streaming_context(self) -> None:
        """Test full workflow using streaming context manager."""
        mock_adapter = Mock()
        mock_adapter.get_candles.return_value = []
        mock_adapter.get_quote.return_value = Mock(last=50500.0)

        with patch("bot_v2.data_providers.coinbase_provider.CoinbaseTickerService") as mock_service:
            with patch("bot_v2.data_providers.coinbase_provider.TickerCache") as mock_cache:
                provider = CoinbaseDataProvider(
                    adapter=mock_adapter, enable_streaming=True
                )

                with provider:
                    # Streaming should be started
                    mock_service.return_value.start.assert_called_once()
                    
                    # Should be able to get data
                    price = provider.get_current_price("BTC")
                    assert price == 50500.0

                # Streaming should be stopped
                mock_service.return_value.stop.assert_called_once()

    def test_error_recovery_workflow(self) -> None:
        """Test error recovery and fallback behavior."""
        mock_adapter = Mock()
        mock_adapter.get_candles.side_effect = Exception("Network error")
        mock_adapter.get_quote.side_effect = Exception("Network error")

        provider = CoinbaseDataProvider(adapter=mock_adapter)

        # Should fall back to mock data
        hist_data = provider.get_historical_data("BTC", "30d", "1d")
        assert isinstance(hist_data, pd.DataFrame)
        assert len(hist_data) > 0

        # Should return default price
        price = provider.get_current_price("BTC")
        assert price == 100.0
