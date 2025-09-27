"""Integration tests for Alpaca paper trading."""

import os
import pytest
from unittest.mock import Mock, patch

from bot.brokers.alpaca import (
    AlpacaClient,
    AlpacaConfig,
    AlpacaExecutor,
    PaperTradingBridge,
    PaperTradingConfig,
)


class TestAlpacaIntegration:
    """Integration tests for Alpaca components."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock Alpaca configuration."""
        return AlpacaConfig(
            api_key="test_key",
            secret_key="test_secret",
            paper_trading=True
        )
    
    @pytest.fixture
    def mock_paper_config(self, mock_config):
        """Mock paper trading configuration."""
        return PaperTradingConfig(
            alpaca_config=mock_config,
            enable_real_time_data=False,  # Disable for tests
            data_symbols=[],
            max_order_value=1000.0,
            max_daily_trades=10,
            log_all_orders=False,
            save_execution_log=False,
        )
    
    def test_alpaca_config_from_env_missing_keys(self):
        """Test config creation with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"):
                AlpacaConfig.from_env()
    
    def test_alpaca_config_from_env_success(self):
        """Test successful config creation from environment."""
        with patch.dict(os.environ, {
            'ALPACA_API_KEY': 'test_key',
            'ALPACA_SECRET_KEY': 'test_secret',
            'ALPACA_PAPER': 'true'
        }):
            config = AlpacaConfig.from_env()
            assert config.api_key == 'test_key'
            assert config.secret_key == 'test_secret'
            assert config.paper_trading is True
    
    @patch('bot.brokers.alpaca.alpaca_client.TradingClient')
    @patch('bot.brokers.alpaca.alpaca_client.StockHistoricalDataClient')
    @patch('bot.brokers.alpaca.alpaca_client.StockDataStream')
    def test_alpaca_client_initialization(self, mock_stream, mock_data, mock_trading, mock_config):
        """Test Alpaca client initialization."""
        # Mock the account response for connection test
        mock_account = Mock()
        mock_account.id = "test_account"
        mock_trading.return_value.get_account.return_value = mock_account
        
        client = AlpacaClient(mock_config)
        
        # Verify clients were created
        mock_trading.assert_called_once()
        mock_data.assert_called_once()
        mock_stream.assert_called_once()
        
        # Verify connection test was called
        mock_trading.return_value.get_account.assert_called_once()
    
    @patch('bot.brokers.alpaca.alpaca_client.TradingClient')
    @patch('bot.brokers.alpaca.alpaca_client.StockHistoricalDataClient')
    @patch('bot.brokers.alpaca.alpaca_client.StockDataStream')
    def test_alpaca_executor_order_validation(self, mock_stream, mock_data, mock_trading, mock_config):
        """Test order validation in executor."""
        # Mock the account response
        mock_account = Mock()
        mock_account.id = "test_account"
        mock_trading.return_value.get_account.return_value = mock_account
        
        client = AlpacaClient(mock_config)
        executor = AlpacaExecutor(client)
        
        # Test invalid order parameters
        result = executor.submit_market_order("", "buy", 100)
        assert not result.success
        assert "Symbol is required" in result.error
        
        result = executor.submit_market_order("AAPL", "invalid", 100)
        assert not result.success
        assert "Side must be 'buy' or 'sell'" in result.error
        
        result = executor.submit_market_order("AAPL", "buy", -100)
        assert not result.success
        assert "Quantity must be positive" in result.error
    
    @patch('bot.brokers.alpaca.alpaca_client.TradingClient')
    @patch('bot.brokers.alpaca.alpaca_client.StockHistoricalDataClient')
    @patch('bot.brokers.alpaca.alpaca_client.StockDataStream')
    def test_paper_trading_bridge_order_validation(self, mock_stream, mock_data, mock_trading, mock_paper_config):
        """Test order validation in paper trading bridge."""
        # Mock the account response
        mock_account = Mock()
        mock_account.id = "test_account"
        mock_trading.return_value.get_account.return_value = mock_account
        
        bridge = PaperTradingBridge(mock_paper_config)
        
        # Test order value limit
        result = bridge.submit_order("AAPL", "buy", 1000, "market")  # Would exceed $1000 limit
        assert not result.success
        assert "exceeds limit" in result.error
        
        # Test valid small order
        with patch.object(bridge, '_get_estimated_price', return_value=1.0):
            with patch.object(bridge.executor, 'submit_market_order') as mock_submit:
                mock_submit.return_value.success = True
                mock_submit.return_value.order_id = "test_order"
                
                result = bridge.submit_order("AAPL", "buy", 10, "market")
                assert result.success
    
    def test_execution_metrics(self):
        """Test execution metrics tracking."""
        from bot.brokers.alpaca.paper_trading_bridge import ExecutionMetrics
        
        metrics = ExecutionMetrics()
        
        # Add successful execution
        metrics.add_execution(True, 150.0, 100, 50.0)
        assert metrics.total_orders == 1
        assert metrics.successful_orders == 1
        assert metrics.failed_orders == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_execution_time_ms == 150.0
        assert metrics.total_volume == 100
        assert metrics.total_notional == 5000.0
        
        # Add failed execution
        metrics.add_execution(False, 200.0, 50, 100.0)
        assert metrics.total_orders == 2
        assert metrics.successful_orders == 1
        assert metrics.failed_orders == 1
        assert metrics.success_rate == 0.5
        assert metrics.avg_execution_time_ms == 175.0  # (150 + 200) / 2
        assert metrics.total_volume == 100  # Only successful orders
        assert metrics.total_notional == 5000.0
    
    @pytest.mark.asyncio
    async def test_paper_trading_bridge_context_manager(self, mock_paper_config):
        """Test paper trading bridge as async context manager."""
        with patch('bot.brokers.alpaca.alpaca_client.TradingClient') as mock_trading:
            with patch('bot.brokers.alpaca.alpaca_client.StockHistoricalDataClient'):
                with patch('bot.brokers.alpaca.alpaca_client.StockDataStream'):
                    # Mock the account response with all required attributes
                    mock_account = Mock()
                    mock_account.id = "test_account"
                    mock_account.account_number = "123456789"
                    mock_account.status = "ACTIVE"
                    mock_account.buying_power = 100000.0
                    mock_account.cash = 50000.0
                    mock_account.portfolio_value = 100000.0
                    mock_account.equity = 100000.0
                    mock_account.pattern_day_trader = False
                    mock_account.daytrade_count = 0
                    mock_account.trading_blocked = False
                    
                    mock_trading.return_value.get_account.return_value = mock_account
                    
                    async with PaperTradingBridge(mock_paper_config) as bridge:
                        assert bridge is not None
                        account = bridge.get_account()
                        assert account.id == "test_account"
    
    def test_market_data_point_creation(self):
        """Test market data point creation from different sources."""
        from bot.brokers.alpaca.alpaca_data import MarketDataPoint
        from datetime import datetime
        
        # Mock Alpaca quote
        mock_quote = Mock()
        mock_quote.symbol = "AAPL"
        mock_quote.timestamp = datetime.now()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.5
        mock_quote.bid_size = 100
        mock_quote.ask_size = 200
        mock_quote.bid_exchange = "NASDAQ"
        mock_quote.ask_exchange = "NASDAQ"
        mock_quote.conditions = []
        
        data_point = MarketDataPoint.from_quote(mock_quote)
        assert data_point.symbol == "AAPL"
        assert data_point.data_type == "quote"
        assert data_point.data["bid_price"] == 150.0
        assert data_point.data["ask_price"] == 150.5
        
        # Mock Alpaca trade
        mock_trade = Mock()
        mock_trade.symbol = "AAPL"
        mock_trade.timestamp = datetime.now()
        mock_trade.price = 150.25
        mock_trade.size = 100
        mock_trade.exchange = "NASDAQ"
        mock_trade.conditions = []
        
        data_point = MarketDataPoint.from_trade(mock_trade)
        assert data_point.symbol == "AAPL"
        assert data_point.data_type == "trade"
        assert data_point.data["price"] == 150.25
        assert data_point.data["size"] == 100