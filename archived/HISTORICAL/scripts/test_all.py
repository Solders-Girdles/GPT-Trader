"""
Comprehensive test suite for minimal trading system.
Goal: 100% test coverage, verify every calculation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from data import DataLoader
from strategy import SimpleMAStrategy
from executor import SimpleExecutor, Position
from ledger import TradeLedger, Transaction, CompletedTrade
from backtest import SimpleBacktest


class TestDataLoader(unittest.TestCase):
    """Test data loading functionality."""
    
    def setUp(self):
        self.loader = DataLoader()
        
    def test_cache_functionality(self):
        """Test that caching works."""
        # First call should cache
        with patch('yfinance.Ticker') as mock_ticker:
            mock_df = pd.DataFrame({
                'Open': [100, 101],
                'High': [102, 103],
                'Low': [99, 100],
                'Close': [101, 102],
                'Volume': [1000, 1100]
            }, index=pd.date_range('2024-01-01', periods=2))
            
            mock_ticker.return_value.history.return_value = mock_df
            
            # First call
            data1 = self.loader.get_data('TEST', datetime(2024, 1, 1), datetime(2024, 1, 2))
            # Second call should use cache
            data2 = self.loader.get_data('TEST', datetime(2024, 1, 1), datetime(2024, 1, 2))
            
            # Should only call YFinance once
            mock_ticker.return_value.history.assert_called_once()
            pd.testing.assert_frame_equal(data1, data2)
            
    def test_clear_cache(self):
        """Test cache clearing."""
        self.loader.cache = {'test': 'data'}
        self.loader.clear_cache()
        self.assertEqual(self.loader.cache, {})


class TestSimpleMAStrategy(unittest.TestCase):
    """Test MA crossover strategy."""
    
    def setUp(self):
        self.strategy = SimpleMAStrategy(fast_period=2, slow_period=3)
        
    def test_generate_signals_crossover(self):
        """Test signal generation on crossovers."""
        # Create data that will produce clear crossovers
        data = pd.DataFrame({
            'Close': [100, 102, 104, 103, 101, 100, 99, 101, 103, 105]
        }, index=pd.date_range('2024-01-01', periods=10))
        
        signals = self.strategy.generate_signals(data)
        
        # Should have signals Series same length as data
        self.assertEqual(len(signals), len(data))
        
        # First few should be 0 (no MA yet)
        self.assertEqual(signals.iloc[0], 0)
        self.assertEqual(signals.iloc[1], 0)
        
        # Should have at least one buy and sell signal
        self.assertIn(1, signals.values)
        self.assertIn(-1, signals.values)
        
    def test_no_close_column_raises(self):
        """Test that missing Close column raises error."""
        data = pd.DataFrame({'Open': [100, 101]})
        
        with self.assertRaises(ValueError) as context:
            self.strategy.generate_signals(data)
        self.assertIn("Close", str(context.exception))
        
    def test_get_parameters(self):
        """Test parameter retrieval."""
        params = self.strategy.get_parameters()
        self.assertEqual(params['fast_period'], 2)
        self.assertEqual(params['slow_period'], 3)


class TestSimpleExecutor(unittest.TestCase):
    """Test position execution."""
    
    def setUp(self):
        self.executor = SimpleExecutor(initial_capital=10000)
        
    def test_buy_signal_creates_position(self):
        """Test that buy signal creates position."""
        action = self.executor.process_signal('AAPL', 1, 100, datetime(2024, 1, 1))
        
        self.assertEqual(action['type'], 'buy')
        self.assertEqual(action['quantity'], 100)  # 10000 / 100
        self.assertIn('AAPL', self.executor.positions)
        self.assertEqual(self.executor.cash, 0)
        
    def test_sell_signal_closes_position(self):
        """Test that sell signal closes position."""
        # First buy
        self.executor.process_signal('AAPL', 1, 100, datetime(2024, 1, 1))
        
        # Then sell
        action = self.executor.process_signal('AAPL', -1, 110, datetime(2024, 1, 2))
        
        self.assertEqual(action['type'], 'sell')
        self.assertEqual(action['realized_pnl'], 1000)  # 100 shares * $10 profit
        self.assertNotIn('AAPL', self.executor.positions)
        self.assertEqual(self.executor.cash, 11000)
        
    def test_hold_signal_does_nothing(self):
        """Test that hold signal doesn't change positions."""
        initial_cash = self.executor.cash
        action = self.executor.process_signal('AAPL', 0, 100, datetime(2024, 1, 1))
        
        self.assertEqual(action['type'], 'hold')
        self.assertEqual(self.executor.cash, initial_cash)
        self.assertEqual(len(self.executor.positions), 0)
        
    def test_portfolio_value_calculation(self):
        """Test portfolio value with positions."""
        # Buy position
        self.executor.process_signal('AAPL', 1, 100, datetime(2024, 1, 1))
        
        # Check value at different price
        value = self.executor.get_portfolio_value({'AAPL': 110})
        self.assertEqual(value, 11000)  # 100 shares * $110
        
    def test_reset(self):
        """Test executor reset."""
        self.executor.process_signal('AAPL', 1, 100, datetime(2024, 1, 1))
        self.executor.reset()
        
        self.assertEqual(self.executor.cash, 10000)
        self.assertEqual(len(self.executor.positions), 0)
        self.assertEqual(len(self.executor.position_history), 0)


class TestTradeLedger(unittest.TestCase):
    """Test trade recording."""
    
    def setUp(self):
        self.ledger = TradeLedger()
        
    def test_record_buy_transaction(self):
        """Test recording a buy."""
        self.ledger.record_transaction(
            datetime(2024, 1, 1), 'AAPL', 'buy', 100, 150
        )
        
        self.assertEqual(len(self.ledger.transactions), 1)
        self.assertEqual(self.ledger.transactions[0].cost, -15000)
        self.assertIn('AAPL', self.ledger.open_positions)
        
    def test_record_sell_creates_completed_trade(self):
        """Test that selling creates completed trade."""
        # Buy
        self.ledger.record_transaction(
            datetime(2024, 1, 1), 'AAPL', 'buy', 100, 150
        )
        
        # Sell
        self.ledger.record_transaction(
            datetime(2024, 1, 5), 'AAPL', 'sell', 100, 160
        )
        
        self.assertEqual(len(self.ledger.completed_trades), 1)
        trade = self.ledger.completed_trades[0]
        self.assertEqual(trade.pnl, 1000)  # 100 * (160 - 150)
        self.assertAlmostEqual(trade.return_pct, 6.67, places=1)  # (10/150) * 100
        self.assertEqual(trade.holding_days, 4)
        
    def test_partial_sell(self):
        """Test selling part of a position."""
        # Buy 100
        self.ledger.record_transaction(
            datetime(2024, 1, 1), 'AAPL', 'buy', 100, 150
        )
        
        # Sell 50
        self.ledger.record_transaction(
            datetime(2024, 1, 2), 'AAPL', 'sell', 50, 160
        )
        
        # Should have 1 completed trade for 50 shares
        self.assertEqual(len(self.ledger.completed_trades), 1)
        self.assertEqual(self.ledger.completed_trades[0].exit_quantity, 50)
        
        # Should still have 50 shares open
        self.assertIn('AAPL', self.ledger.open_positions)
        remaining = self.ledger.open_positions['AAPL'][0]['quantity']
        self.assertEqual(remaining, 50)
        
    def test_calculate_statistics(self):
        """Test statistics calculation."""
        # Make some trades
        self.ledger.record_transaction(datetime(2024, 1, 1), 'AAPL', 'buy', 100, 150)
        self.ledger.record_transaction(datetime(2024, 1, 2), 'AAPL', 'sell', 100, 160)
        self.ledger.record_transaction(datetime(2024, 1, 3), 'AAPL', 'buy', 100, 155)
        self.ledger.record_transaction(datetime(2024, 1, 4), 'AAPL', 'sell', 100, 152)
        
        stats = self.ledger.calculate_statistics()
        
        self.assertEqual(stats['total_transactions'], 4)
        self.assertEqual(stats['total_completed_trades'], 2)
        self.assertEqual(stats['total_pnl'], 700)  # 1000 - 300
        self.assertEqual(stats['win_rate'], 50)  # 1 win, 1 loss


class TestSimpleBacktest(unittest.TestCase):
    """Test backtesting engine."""
    
    def setUp(self):
        self.backtest = SimpleBacktest(initial_capital=10000)
        
    @patch('data.DataLoader.get_data')
    def test_full_backtest_run(self, mock_get_data):
        """Test complete backtest execution."""
        # Create mock data
        dates = pd.date_range('2024-01-01', periods=50)
        prices = 100 + np.sin(np.arange(50) * 0.3) * 10  # Oscillating price
        
        mock_data = pd.DataFrame({
            'Open': prices,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices,
            'Volume': [1000000] * 50
        }, index=dates)
        
        mock_get_data.return_value = mock_data
        
        # Run backtest
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)
        results = self.backtest.run(
            strategy, 'TEST',
            datetime(2024, 1, 1),
            datetime(2024, 2, 19)
        )
        
        # Check results structure
        self.assertIsNotNone(results)
        self.assertEqual(results.symbol, 'TEST')
        self.assertEqual(results.initial_capital, 10000)
        self.assertGreater(results.total_transactions, 0)
        self.assertIsInstance(results.portfolio_values, pd.Series)
        
    def test_calculate_sharpe(self):
        """Test Sharpe ratio calculation."""
        returns = pd.Series([0.01, -0.005, 0.008, 0.002, -0.003])
        sharpe = self.backtest._calculate_sharpe(returns)
        
        # Should be a reasonable number
        self.assertIsInstance(sharpe, float)
        self.assertFalse(np.isnan(sharpe))
        
    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        values = pd.Series([100, 110, 105, 95, 100, 90, 95])
        drawdown = self.backtest._calculate_max_drawdown(values)
        
        # Max drawdown should be from 110 to 90 = -18.18%
        self.assertAlmostEqual(drawdown, -18.18, places=1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    @patch('yfinance.Ticker')
    def test_end_to_end_workflow(self, mock_ticker):
        """Test complete workflow from data to results."""
        # Setup mock data with clear crossovers
        dates = pd.date_range('2024-01-01', periods=30)
        # Create prices that will generate MA crossovers
        prices = []
        for i in range(30):
            if i < 10:
                prices.append(100 - i)  # Falling
            elif i < 20:
                prices.append(90 + (i - 10) * 2)  # Rising
            else:
                prices.append(110 - (i - 20))  # Falling again
        
        mock_df = pd.DataFrame({
            'Open': prices,
            'High': [p + 1 for p in prices],
            'Low': [p - 1 for p in prices],
            'Close': prices,
            'Volume': [1000000] * 30
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = mock_df
        
        # Run complete workflow
        loader = DataLoader()
        data = loader.get_data('TEST', datetime(2024, 1, 1), datetime(2024, 1, 30))
        
        strategy = SimpleMAStrategy(fast_period=2, slow_period=5)  # Shorter periods for test data
        signals = strategy.generate_signals(data)
        
        executor = SimpleExecutor(10000)
        ledger = TradeLedger()
        
        for date, row in data.iterrows():
            signal = signals.loc[date]
            price = row['Close']
            
            action = executor.process_signal('TEST', signal, price, date)
            
            if action['type'] in ['buy', 'sell']:
                ledger.record_transaction(
                    date, 'TEST', action['type'],
                    action['quantity'], price
                )
        
        # Verify we got some trades
        self.assertGreater(len(ledger.transactions), 0)
        
        # Verify final portfolio value makes sense
        final_value = executor.get_portfolio_value({'TEST': prices[-1]})
        self.assertGreater(final_value, 0)


def run_all_tests():
    """Run all tests and report coverage."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleMAStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleExecutor))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeLedger))
    suite.addTests(loader.loadTestsFromTestCase(TestSimpleBacktest))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Report results
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)