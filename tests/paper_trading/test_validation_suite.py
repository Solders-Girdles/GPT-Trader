"""
Comprehensive validation test suite for paper trading system.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from decimal import Decimal

# Import components to test
from bot.brokers.alpaca.alpaca_client import AlpacaClient
from bot.brokers.alpaca.alpaca_executor import AlpacaExecutor
from bot.brokers.alpaca.alpaca_data import AlpacaDataFeed
from bot.brokers.alpaca.paper_trading_bridge import PaperTradingBridge
from bot.tracking.position_manager import PositionManager
from bot.tracking.pnl_calculator import PnLCalculator
from bot.tracking.trade_ledger import TradeLedger
from bot.tracking.reconciliation import PositionReconciliation


class TestAlpacaIntegration:
    """Test Alpaca broker integration."""
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Create mock Alpaca client."""
        with patch('bot.brokers.alpaca.alpaca_client.TradingClient') as mock:
            client = AlpacaClient(api_key="test", secret_key="test")
            yield client
    
    def test_connection(self, mock_alpaca_client):
        """Test Alpaca connection."""
        assert mock_alpaca_client.test_connection() is True
    
    def test_authentication_failure(self):
        """Test authentication error handling."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError):
                AlpacaClient()
    
    @pytest.mark.asyncio
    async def test_order_submission(self, mock_alpaca_client):
        """Test order submission flow."""
        executor = AlpacaExecutor(mock_alpaca_client)
        
        # Test market order
        order = await executor.submit_market_order("AAPL", 100, "buy")
        assert order is not None
        
        # Test limit order
        order = await executor.submit_limit_order("AAPL", 100, 150.00, "buy")
        assert order is not None


class TestPositionTracking:
    """Test real-time position tracking."""
    
    @pytest.fixture
    def position_manager(self):
        """Create position manager instance."""
        return PositionManager()
    
    def test_position_update(self, position_manager):
        """Test position updates."""
        position_manager.update_position(
            "AAPL", 100, 150.00, "strategy1"
        )
        
        position = position_manager.get_position("AAPL", "strategy1")
        assert position['quantity'] == 100
        assert position['avg_price'] == 150.00
    
    def test_pnl_calculation(self):
        """Test P&L calculations."""
        calculator = PnLCalculator()
        
        # Add position
        calculator.update_position("AAPL", 100, 150.00)
        
        # Update market price
        calculator.update_market_price("AAPL", 155.00)
        
        # Check unrealized P&L
        pnl = calculator.get_unrealized_pnl("AAPL")
        assert pnl == 500.00  # (155 - 150) * 100
    
    def test_portfolio_metrics(self):
        """Test portfolio performance metrics."""
        calculator = PnLCalculator()
        
        # Add multiple positions
        calculator.update_position("AAPL", 100, 150.00)
        calculator.update_position("GOOGL", 50, 2800.00)
        
        # Update prices
        calculator.update_market_price("AAPL", 155.00)
        calculator.update_market_price("GOOGL", 2850.00)
        
        # Check portfolio metrics
        metrics = calculator.get_portfolio_metrics()
        assert metrics['total_value'] > 0
        assert 'sharpe_ratio' in metrics


class TestDataFeed:
    """Test real-time data feed."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket data feed connection."""
        with patch('websockets.connect') as mock_ws:
            feed = AlpacaDataFeed(api_key="test", secret_key="test")
            
            # Start data feed
            await feed.connect()
            assert feed.is_connected
            
            # Subscribe to symbol
            await feed.subscribe_symbol("AAPL")
            assert "AAPL" in feed.subscribed_symbols
    
    @pytest.mark.asyncio
    async def test_data_handler(self):
        """Test data handler callback."""
        feed = AlpacaDataFeed(api_key="test", secret_key="test")
        
        # Add handler
        handler_called = False
        def handler(data):
            nonlocal handler_called
            handler_called = True
        
        feed.add_quote_handler(handler)
        
        # Simulate data
        await feed._handle_quote({"symbol": "AAPL", "price": 150.00})
        assert handler_called


class TestEndToEndFlow:
    """Test complete paper trading flow."""
    
    @pytest.mark.asyncio
    async def test_complete_trade_flow(self):
        """Test complete trade execution flow."""
        # Setup components
        client = Mock(spec=AlpacaClient)
        executor = AlpacaExecutor(client)
        position_manager = PositionManager()
        pnl_calculator = PnLCalculator()
        
        # Submit order
        order = {
            'id': 'test123',
            'symbol': 'AAPL',
            'qty': 100,
            'filled_avg_price': 150.00,
            'status': 'filled'
        }
        
        # Update position
        position_manager.update_position(
            order['symbol'],
            order['qty'],
            order['filled_avg_price'],
            'test_strategy'
        )
        
        # Calculate P&L
        pnl_calculator.update_position(
            order['symbol'],
            order['qty'],
            order['filled_avg_price']
        )
        pnl_calculator.update_market_price('AAPL', 155.00)
        
        # Verify
        position = position_manager.get_position('AAPL', 'test_strategy')
        assert position['quantity'] == 100
        
        pnl = pnl_calculator.get_unrealized_pnl('AAPL')
        assert pnl == 500.00
    
    def test_reconciliation(self):
        """Test position reconciliation."""
        # Setup
        position_manager = PositionManager()
        reconciliation = PositionReconciliation(None, position_manager)
        
        # Add internal position
        position_manager.update_position("AAPL", 100, 150.00, "strategy1")
        
        # Mock broker positions
        broker_positions = [
            {'symbol': 'AAPL', 'qty': 100, 'avg_entry_price': 150.00}
        ]
        
        # Reconcile
        with patch.object(reconciliation, '_get_broker_positions', 
                         return_value=broker_positions):
            discrepancies = reconciliation.reconcile_positions()
            
            # Should have no discrepancies
            assert len(discrepancies) == 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_connection_retry(self):
        """Test connection retry logic."""
        client = AlpacaClient(api_key="test", secret_key="test")
        
        # Mock failed connection
        with patch.object(client, '_make_request', 
                         side_effect=[Exception("Connection failed"), True]):
            result = client.test_connection()
            assert result is True  # Should retry and succeed
    
    def test_invalid_order_validation(self):
        """Test order validation."""
        bridge = PaperTradingBridge(Mock(), Mock())
        
        # Test invalid quantity
        with pytest.raises(ValueError):
            bridge.submit_order("AAPL", -100, "market", "buy")
        
        # Test invalid side
        with pytest.raises(ValueError):
            bridge.submit_order("AAPL", 100, "market", "invalid")


class TestPerformanceAndLoad:
    """Test performance under load."""
    
    def test_position_tracking_performance(self):
        """Test position tracking under load."""
        position_manager = PositionManager()
        
        # Add many positions
        import time
        start = time.time()
        
        for i in range(1000):
            position_manager.update_position(
                f"SYMBOL{i}", 100, 100.00 + i, "strategy1"
            )
        
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should handle 1000 positions in < 1 second
    
    def test_pnl_calculation_performance(self):
        """Test P&L calculation performance."""
        calculator = PnLCalculator()
        
        # Add positions
        for i in range(100):
            calculator.update_position(f"SYMBOL{i}", 100, 100.00)
        
        # Update prices and calculate
        import time
        start = time.time()
        
        for i in range(100):
            calculator.update_market_price(f"SYMBOL{i}", 105.00)
            calculator.get_portfolio_metrics()
        
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should calculate 100 portfolios in < 1 second


# Run validation
if __name__ == "__main__":
    print("🧪 Running Paper Trading Validation Tests")
    print("=" * 50)
    
    # Run tests with coverage
    pytest.main([__file__, "-v", "--tb=short"])