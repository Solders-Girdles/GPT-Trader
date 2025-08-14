"""
Comprehensive unit tests for Portfolio Backtest Engine.

Tests portfolio engine validation, trade execution simulation,
performance metrics, multi-asset backtesting, and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile

from bot.backtest.engine_portfolio import (
    BacktestData,
    BacktestConfig,
    BacktestEngine,
    validate_ohlc,
    prepare_backtest_data,
    run_backtest,
    _iter_progress,
    _warn,
    _read_universe_csv
)
from bot.strategy.base import Strategy
from bot.portfolio.allocator import PortfolioRules


class TestStrategyImpl(Strategy):
    """Test strategy implementation."""
    
    name = "test_strategy"
    supports_short = False
    
    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate simple test signals."""
        df = bars.copy()
        df['signal'] = 0
        
        # Simple MA crossover
        if len(df) >= 20:
            sma_fast = df['Close'].rolling(5).mean()
            sma_slow = df['Close'].rolling(20).mean()
            df.loc[sma_fast > sma_slow, 'signal'] = 1
        
        df['atr'] = df['Close'].rolling(14).std()
        return df[['signal', 'atr']]


class TestBacktestData:
    """Test BacktestData dataclass."""
    
    def test_backtest_data_creation(self):
        """Test BacktestData initialization."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        
        data_map = {}
        for symbol in symbols:
            data_map[symbol] = pd.DataFrame({
                "Open": np.random.uniform(95, 105, 100),
                "High": np.random.uniform(100, 110, 100),
                "Low": np.random.uniform(90, 100, 100),
                "Close": np.random.uniform(95, 105, 100),
                "Volume": np.random.uniform(1000000, 10000000, 100)
            }, index=dates)
        
        regime_ok = pd.Series([True] * 100, index=dates)
        
        backtest_data = BacktestData(
            symbols=symbols,
            data_map=data_map,
            regime_ok=regime_ok,
            dates_idx=dates
        )
        
        assert backtest_data.symbols == symbols
        assert len(backtest_data.data_map) == 3
        assert backtest_data.dates_idx.equals(dates)
        assert backtest_data.regime_ok is not None


class TestValidateOHLC:
    """Test OHLC validation function."""
    
    @pytest.fixture
    def valid_ohlc(self):
        """Create valid OHLC data."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        return pd.DataFrame({
            "Open": [100 + i * 0.1 for i in range(30)],
            "High": [101 + i * 0.1 for i in range(30)],
            "Low": [99 + i * 0.1 for i in range(30)],
            "Close": [100.5 + i * 0.1 for i in range(30)],
            "Volume": [1000000] * 30
        }, index=dates)
    
    def test_validate_valid_data(self, valid_ohlc):
        """Test validation with valid data."""
        result = validate_ohlc(valid_ohlc, "TEST")
        assert result is not None
        pd.testing.assert_frame_equal(result, valid_ohlc)
    
    def test_validate_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="empty DataFrame"):
            validate_ohlc(empty_df, "TEST")
    
    def test_validate_non_datetime_index(self):
        """Test validation with non-DatetimeIndex."""
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [101, 102, 103],
            "Low": [99, 100, 101],
            "Close": [100.5, 101.5, 102.5]
        })
        
        with pytest.raises(TypeError, match="index must be DatetimeIndex"):
            validate_ohlc(df, "TEST")
    
    def test_validate_unsorted_index(self):
        """Test validation with unsorted DatetimeIndex."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        shuffled_dates = dates[[2, 0, 4, 1, 3]]
        
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5]
        }, index=shuffled_dates)
        
        with pytest.raises(ValueError, match="not sorted ascending"):
            validate_ohlc(df, "TEST")
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5]
        }, index=dates)
        
        with pytest.raises(KeyError, match="missing required columns"):
            validate_ohlc(df, "TEST")
    
    def test_validate_nan_values(self):
        """Test validation with NaN values."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        
        df = pd.DataFrame({
            "Open": [100, np.nan, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5]
        }, index=dates)
        
        with pytest.raises(ValueError, match="NaNs in OHLC"):
            validate_ohlc(df, "TEST")
    
    def test_validate_invalid_ohlc_bounds(self):
        """Test validation with invalid OHLC relationships."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        
        # High < Close
        df = pd.DataFrame({
            "Open": [100, 101, 102, 103, 104],
            "High": [99, 102, 103, 104, 105],  # First high < close
            "Low": [98, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5]
        }, index=dates)
        
        with pytest.raises(ValueError, match="invalid OHLC bounds"):
            validate_ohlc(df, "TEST")
    
    def test_validate_long_gaps_warning(self, valid_ohlc):
        """Test validation with long gaps in dates."""
        # Add a gap
        dates_with_gap = valid_ohlc.index[:10].append(
            valid_ohlc.index[10:] + timedelta(days=20)
        )
        df_with_gap = valid_ohlc.copy()
        df_with_gap.index = dates_with_gap
        
        # Should pass but may warn about gaps
        result = validate_ohlc(df_with_gap, "TEST", strict_mode=False)
        assert result is not None


class TestLoadUniverse:
    """Test universe loading functionality."""
    
    def test_load_universe_from_csv(self):
        """Test loading universe from CSV file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("symbol\n")
            f.write("AAPL\n")
            f.write("GOOGL\n")
            f.write("MSFT\n")
            csv_path = f.name
        
        try:
            symbols = load_universe(csv_path)
            assert symbols == ["AAPL", "GOOGL", "MSFT"]
        finally:
            Path(csv_path).unlink()
    
    def test_load_universe_from_list(self):
        """Test loading universe from list."""
        symbols = load_universe(["AAPL", "GOOGL", "MSFT"])
        assert symbols == ["AAPL", "GOOGL", "MSFT"]
    
    def test_load_universe_from_string(self):
        """Test loading universe from string."""
        symbols = load_universe("AAPL")
        assert symbols == ["AAPL"]
    
    def test_load_universe_nonexistent_file(self):
        """Test loading universe from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_universe("/nonexistent/path.csv")


class TestPrepareData:
    """Test data preparation functionality."""
    
    @pytest.fixture
    def mock_data_source(self):
        """Create mock data source."""
        mock = MagicMock()
        
        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        sample_data = pd.DataFrame({
            "Open": np.random.uniform(95, 105, 100),
            "High": np.random.uniform(100, 110, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(95, 105, 100),
            "Adj Close": np.random.uniform(95, 105, 100),
            "Volume": np.random.uniform(1000000, 10000000, 100)
        }, index=dates)
        
        mock.fetch.return_value = sample_data
        return mock
    
    @patch('bot.backtest.engine_portfolio.YFinanceSource')
    def test_prepare_data_basic(self, mock_yfinance_class, mock_data_source):
        """Test basic data preparation."""
        mock_yfinance_class.return_value = mock_data_source
        
        symbols = ["AAPL", "GOOGL"]
        start_date = "2024-01-01"
        end_date = "2024-03-31"
        
        backtest_data = prepare_data(
            symbols=symbols,
            start=start_date,
            end=end_date
        )
        
        assert backtest_data.symbols == symbols
        assert len(backtest_data.data_map) == 2
        assert "AAPL" in backtest_data.data_map
        assert "GOOGL" in backtest_data.data_map
        assert backtest_data.dates_idx is not None
    
    @patch('bot.backtest.engine_portfolio.YFinanceSource')
    def test_prepare_data_with_validation(self, mock_yfinance_class, mock_data_source):
        """Test data preparation with validation."""
        mock_yfinance_class.return_value = mock_data_source
        
        symbols = ["AAPL"]
        
        backtest_data = prepare_data(
            symbols=symbols,
            start="2024-01-01",
            end="2024-03-31",
            validate=True
        )
        
        assert backtest_data is not None
        assert len(backtest_data.data_map) == 1


class TestRunStrategyBacktest:
    """Test single strategy backtesting."""
    
    @pytest.fixture
    def sample_backtest_data(self):
        """Create sample backtest data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        
        # Create realistic price data
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            "Open": prices * (1 + np.random.uniform(-0.005, 0.005, 100)),
            "High": prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
            "Low": prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
            "Close": prices,
            "Volume": np.random.uniform(1000000, 10000000, 100)
        }, index=dates)
        
        return BacktestData(
            symbols=["TEST"],
            data_map={"TEST": data},
            regime_ok=pd.Series([True] * 100, index=dates),
            dates_idx=dates
        )
    
    def test_run_strategy_backtest_basic(self, sample_backtest_data):
        """Test basic strategy backtest."""
        strategy = TestStrategyImpl()
        
        metrics = run_strategy_backtest(
            data=sample_backtest_data,
            strategy=strategy,
            symbol="TEST",
            capital=10000
        )
        
        assert metrics is not None
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
    
    def test_run_strategy_backtest_with_rules(self, sample_backtest_data):
        """Test strategy backtest with portfolio rules."""
        strategy = TestStrategyImpl()
        rules = PortfolioRules(
            max_position_size=0.2,
            max_portfolio_size=0.95,
            stop_loss=0.05,
            take_profit=0.10
        )
        
        metrics = run_strategy_backtest(
            data=sample_backtest_data,
            strategy=strategy,
            symbol="TEST",
            capital=10000,
            rules=rules
        )
        
        assert metrics is not None
    
    def test_run_strategy_backtest_no_signals(self, sample_backtest_data):
        """Test backtest with strategy generating no signals."""
        
        class NoSignalStrategy(Strategy):
            name = "no_signal"
            supports_short = False
            
            def generate_signals(self, bars):
                df = bars.copy()
                df['signal'] = 0  # No signals
                return df[['signal']]
        
        strategy = NoSignalStrategy()
        
        metrics = run_strategy_backtest(
            data=sample_backtest_data,
            strategy=strategy,
            symbol="TEST",
            capital=10000
        )
        
        # Should still return metrics even with no trades
        assert metrics is not None
        assert metrics.get("total_return", 0) == 0
    
    def test_run_strategy_backtest_with_commissions(self, sample_backtest_data):
        """Test backtest with commission costs."""
        strategy = TestStrategyImpl()
        
        metrics = run_strategy_backtest(
            data=sample_backtest_data,
            strategy=strategy,
            symbol="TEST",
            capital=10000,
            commission=0.001  # 0.1% commission
        )
        
        assert metrics is not None


class TestRunPortfolioBacktest:
    """Test portfolio-level backtesting."""
    
    @pytest.fixture
    def multi_asset_data(self):
        """Create multi-asset backtest data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        data_map = {}
        np.random.seed(42)
        
        for i, symbol in enumerate(symbols):
            # Different return profiles for each asset
            returns = np.random.normal(0.001 + i * 0.0002, 0.02 - i * 0.002, 100)
            prices = (100 + i * 50) * np.exp(np.cumsum(returns))
            
            data_map[symbol] = pd.DataFrame({
                "Open": prices * (1 + np.random.uniform(-0.005, 0.005, 100)),
                "High": prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
                "Low": prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
                "Close": prices,
                "Volume": np.random.uniform(1000000, 10000000, 100)
            }, index=dates)
        
        return BacktestData(
            symbols=symbols,
            data_map=data_map,
            regime_ok=pd.Series([True] * 100, index=dates),
            dates_idx=dates
        )
    
    def test_run_portfolio_backtest_basic(self, multi_asset_data):
        """Test basic portfolio backtest."""
        strategy = TestStrategyImpl()
        
        portfolio_metrics, asset_metrics = run_portfolio_backtest(
            data=multi_asset_data,
            strategy=strategy,
            capital=100000
        )
        
        assert portfolio_metrics is not None
        assert asset_metrics is not None
        assert len(asset_metrics) == 3  # One for each asset
        
        # Check portfolio metrics
        assert "total_return" in portfolio_metrics
        assert "sharpe_ratio" in portfolio_metrics
        assert "max_drawdown" in portfolio_metrics
    
    def test_run_portfolio_backtest_with_allocation(self, multi_asset_data):
        """Test portfolio backtest with custom allocation rules."""
        strategy = TestStrategyImpl()
        rules = PortfolioRules(
            max_position_size=0.3,
            max_portfolio_size=0.95,
            min_position_size=0.05
        )
        
        portfolio_metrics, asset_metrics = run_portfolio_backtest(
            data=multi_asset_data,
            strategy=strategy,
            capital=100000,
            rules=rules
        )
        
        assert portfolio_metrics is not None
        assert asset_metrics is not None
    
    def test_run_portfolio_backtest_single_asset(self):
        """Test portfolio backtest with single asset."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="D")
        
        data = pd.DataFrame({
            "Open": np.random.uniform(95, 105, 50),
            "High": np.random.uniform(100, 110, 50),
            "Low": np.random.uniform(90, 100, 50),
            "Close": np.random.uniform(95, 105, 50),
            "Volume": [1000000] * 50
        }, index=dates)
        
        backtest_data = BacktestData(
            symbols=["TEST"],
            data_map={"TEST": data},
            regime_ok=None,
            dates_idx=dates
        )
        
        strategy = TestStrategyImpl()
        
        portfolio_metrics, asset_metrics = run_portfolio_backtest(
            data=backtest_data,
            strategy=strategy,
            capital=10000
        )
        
        assert portfolio_metrics is not None
        assert len(asset_metrics) == 1


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_iter_progress_with_tqdm(self):
        """Test progress iterator with tqdm available."""
        items = list(range(10))
        
        result = list(_iter_progress(items, desc="test"))
        assert result == items
    
    @patch('bot.backtest.engine_portfolio.logger')
    def test_warn_function(self, mock_logger):
        """Test warning function."""
        _warn("Test warning message")
        mock_logger.warning.assert_called_once_with("Test warning message")


class TestIntegrationScenarios:
    """Integration tests for complete backtest scenarios."""
    
    @pytest.fixture
    def complete_market_data(self):
        """Create complete market scenario."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        data_map = {}
        np.random.seed(42)
        
        for symbol in symbols:
            # Create correlated but distinct price series
            market_return = np.random.normal(0.0005, 0.015, len(dates))
            idiosyncratic_return = np.random.normal(0, 0.01, len(dates))
            total_return = market_return * 0.7 + idiosyncratic_return * 0.3
            
            prices = 100 * np.exp(np.cumsum(total_return))
            
            data_map[symbol] = pd.DataFrame({
                "Open": prices * (1 + np.random.uniform(-0.003, 0.003, len(dates))),
                "High": prices * (1 + np.abs(np.random.uniform(0, 0.008, len(dates)))),
                "Low": prices * (1 - np.abs(np.random.uniform(0, 0.008, len(dates)))),
                "Close": prices,
                "Volume": np.random.uniform(5000000, 20000000, len(dates))
            }, index=dates)
        
        return BacktestData(
            symbols=symbols,
            data_map=data_map,
            regime_ok=None,
            dates_idx=dates
        )
    
    def test_full_year_portfolio_backtest(self, complete_market_data):
        """Test full year portfolio backtest with multiple assets."""
        strategy = TestStrategyImpl()
        rules = PortfolioRules(
            max_position_size=0.25,
            max_portfolio_size=0.95,
            min_position_size=0.05,
            stop_loss=0.03,
            take_profit=0.08
        )
        
        portfolio_metrics, asset_metrics = run_portfolio_backtest(
            data=complete_market_data,
            strategy=strategy,
            capital=1000000,
            rules=rules,
            commission=0.001
        )
        
        # Verify portfolio metrics
        assert portfolio_metrics is not None
        assert -1 <= portfolio_metrics.get("total_return", 0) <= 2  # Reasonable return range
        assert portfolio_metrics.get("max_drawdown", 0) <= 0  # Drawdown is negative
        
        # Verify individual asset metrics
        assert len(asset_metrics) == 5
        for symbol, metrics in asset_metrics.items():
            assert symbol in complete_market_data.symbols
            assert "total_return" in metrics
    
    def test_stress_test_large_universe(self):
        """Stress test with large universe of assets."""
        dates = pd.date_range(start="2024-01-01", periods=252, freq="D")  # One year
        symbols = [f"STOCK_{i:03d}" for i in range(50)]  # 50 stocks
        
        data_map = {}
        np.random.seed(42)
        
        for symbol in symbols:
            returns = np.random.normal(0.0003, 0.02, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            data_map[symbol] = pd.DataFrame({
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1000000] * len(dates)
            }, index=dates)
        
        backtest_data = BacktestData(
            symbols=symbols,
            data_map=data_map,
            regime_ok=None,
            dates_idx=dates
        )
        
        strategy = TestStrategyImpl()
        
        # Should handle large universe efficiently
        portfolio_metrics, asset_metrics = run_portfolio_backtest(
            data=backtest_data,
            strategy=strategy,
            capital=10000000
        )
        
        assert portfolio_metrics is not None
        assert len(asset_metrics) == 50