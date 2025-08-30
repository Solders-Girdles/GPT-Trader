"""
Comprehensive unit tests for Portfolio Backtest Engine.

Tests portfolio engine validation, trade execution simulation,
performance metrics, multi-asset backtesting, and edge cases.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from bot.backtest.engine_portfolio import (
    BacktestConfig,
    BacktestData,
    BacktestEngine,
    _iter_progress,
    _warn,
    prepare_backtest_data,
    validate_ohlc,
)
from bot.strategy.base import Strategy


class TestStrategyImpl(Strategy):
    """Test strategy implementation."""

    name = "test_strategy"
    supports_short = False

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate simple test signals."""
        df = bars.copy()
        df["signal"] = 0

        # Simple MA crossover
        if len(df) >= 20:
            sma_fast = df["close"].rolling(5).mean()
            sma_slow = df["close"].rolling(20).mean()
            df.loc[sma_fast > sma_slow, "signal"] = 1

        df["atr"] = df["close"].rolling(14).std()
        return df[["signal", "atr"]]


class TestBacktestData:
    """Test BacktestData dataclass."""

    def test_backtest_data_creation(self):
        """Test BacktestData initialization."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

        data_map = {}
        for symbol in symbols:
            data_map[symbol] = pd.DataFrame(
                {
                    "Open": np.random.uniform(95, 105, 100),
                    "High": np.random.uniform(100, 110, 100),
                    "Low": np.random.uniform(90, 100, 100),
                    "Close": np.random.uniform(95, 105, 100),
                    "Volume": np.random.uniform(1000000, 10000000, 100),
                },
                index=dates,
            )

        regime_ok = pd.Series([True] * 100, index=dates)

        backtest_data = BacktestData(
            symbols=symbols, data_map=data_map, regime_ok=regime_ok, dates_idx=dates
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
        return pd.DataFrame(
            {
                "Open": [100 + i * 0.1 for i in range(30)],
                "High": [101 + i * 0.1 for i in range(30)],
                "Low": [99 + i * 0.1 for i in range(30)],
                "Close": [100.5 + i * 0.1 for i in range(30)],
                "Volume": [1000000] * 30,
            },
            index=dates,
        )

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
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Close": [100.5, 101.5, 102.5],
            }
        )

        with pytest.raises(TypeError, match="index must be DatetimeIndex"):
            validate_ohlc(df, "TEST")

    def test_validate_unsorted_index(self):
        """Test validation with unsorted DatetimeIndex."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")
        shuffled_dates = dates[[2, 0, 4, 1, 3]]

        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            },
            index=shuffled_dates,
        )

        with pytest.raises(ValueError, match="not sorted ascending"):
            validate_ohlc(df, "TEST")

    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")

        df = pd.DataFrame(
            {"Open": [100, 101, 102, 103, 104], "Close": [100.5, 101.5, 102.5, 103.5, 104.5]},
            index=dates,
        )

        with pytest.raises(KeyError, match="missing required columns"):
            validate_ohlc(df, "TEST")

    def test_validate_nan_values(self):
        """Test validation with NaN values."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")

        df = pd.DataFrame(
            {
                "Open": [100, np.nan, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="NaNs in OHLC"):
            validate_ohlc(df, "TEST")

    def test_validate_invalid_ohlc_bounds(self):
        """Test validation with invalid OHLC relationships."""
        dates = pd.date_range(start="2024-01-01", periods=5, freq="D")

        # High < Close
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [99, 102, 103, 104, 105],  # First high < close
                "Low": [98, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="invalid OHLC bounds"):
            validate_ohlc(df, "TEST")

    def test_validate_long_gaps_warning(self, valid_ohlc):
        """Test validation with long gaps in dates."""
        # Add a gap
        dates_with_gap = valid_ohlc.index[:10].append(valid_ohlc.index[10:] + timedelta(days=20))
        df_with_gap = valid_ohlc.copy()
        df_with_gap.index = dates_with_gap

        # Should pass but may warn about gaps
        result = validate_ohlc(df_with_gap, "TEST", strict_mode=False)
        assert result is not None


class TestBacktestEngine:
    """Test BacktestEngine class."""

    @pytest.fixture
    def backtest_config(self):
        """Create test backtest config."""
        return BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
            initial_capital=10000,
            commission_rate=0.001,
        )

    @pytest.fixture
    def test_data(self):
        """Create test market data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, 100)
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "Open": prices * (1 + np.random.uniform(-0.005, 0.005, 100)),
                "High": prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
                "Low": prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
                "Close": prices,
                "Volume": np.random.uniform(1000000, 10000000, 100),
            },
            index=dates,
        )

    def test_backtest_engine_creation(self, backtest_config):
        """Test creating backtest engine."""
        engine = BacktestEngine(backtest_config)
        assert engine.config == backtest_config

    def test_backtest_engine_empty_data(self, backtest_config):
        """Test engine with empty data."""
        engine = BacktestEngine(backtest_config)
        strategy = TestStrategyImpl()
        empty_data = pd.DataFrame()

        result = engine.run_backtest(strategy, empty_data)

        assert result is not None
        assert "metrics" in result
        metrics = result["metrics"]
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["total_return"] == 0.0

    def test_backtest_engine_with_data(self, backtest_config, test_data):
        """Test engine with actual data."""
        engine = BacktestEngine(backtest_config)
        strategy = TestStrategyImpl()

        result = engine.run_backtest(strategy, test_data)

        assert result is not None
        assert "metrics" in result
        metrics = result["metrics"]
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["max_drawdown"], float)


class TestPrepareBacktestData:
    """Test data preparation functionality."""

    @pytest.fixture
    def mock_data_source(self):
        """Create mock data source."""
        mock = MagicMock()

        # Create sample data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        sample_data = pd.DataFrame(
            {
                "Open": np.random.uniform(95, 105, 100),
                "High": np.random.uniform(100, 110, 100),
                "Low": np.random.uniform(90, 100, 100),
                "Close": np.random.uniform(95, 105, 100),
                "Adj Close": np.random.uniform(95, 105, 100),
                "Volume": np.random.uniform(1000000, 10000000, 100),
            },
            index=dates,
        )

        mock.get_daily_bars.return_value = sample_data
        return mock

    @patch("bot.backtest.engine_portfolio.YFinanceSource")
    def test_prepare_backtest_data_basic(self, mock_yfinance_class, mock_data_source):
        """Test basic data preparation."""
        mock_yfinance_class.return_value = mock_data_source

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)

        backtest_data = prepare_backtest_data(
            symbol="AAPL",
            symbol_list_csv=None,
            start=start_date,
            end=end_date,
        )

        assert backtest_data.symbols == ["AAPL"]
        assert len(backtest_data.data_map) == 1
        assert "AAPL" in backtest_data.data_map
        assert backtest_data.dates_idx is not None

    @patch("bot.backtest.engine_portfolio.YFinanceSource")
    def test_prepare_backtest_data_with_symbols_list(self, mock_yfinance_class, mock_data_source):
        """Test data preparation with symbols list."""
        mock_yfinance_class.return_value = mock_data_source

        symbols = ["AAPL", "GOOGL"]
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 3, 31)

        backtest_data = prepare_backtest_data(
            symbol=None, symbol_list_csv=None, start=start_date, end=end_date, symbols=symbols
        )

        assert backtest_data.symbols == symbols
        assert len(backtest_data.data_map) == 2
        assert backtest_data.dates_idx is not None


class TestUtilityFunctions:
    """Test utility functions."""

    def test_iter_progress_with_tqdm(self):
        """Test progress iterator with tqdm available."""
        items = list(range(10))

        result = list(_iter_progress(items, desc="test"))
        assert result == items

    @patch("bot.backtest.engine_portfolio.logger")
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

            data_map[symbol] = pd.DataFrame(
                {
                    "Open": prices * (1 + np.random.uniform(-0.003, 0.003, len(dates))),
                    "High": prices * (1 + np.abs(np.random.uniform(0, 0.008, len(dates)))),
                    "Low": prices * (1 - np.abs(np.random.uniform(0, 0.008, len(dates)))),
                    "Close": prices,
                    "Volume": np.random.uniform(5000000, 20000000, len(dates)),
                },
                index=dates,
            )

        return BacktestData(symbols=symbols, data_map=data_map, regime_ok=None, dates_idx=dates)

    def test_backtest_engine_with_strategy(self, complete_market_data):
        """Test backtest engine with a real strategy and multiple assets."""
        # Test single asset from the complete data
        symbol = "AAPL"
        data = complete_market_data.data_map[symbol]

        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
        )

        engine = BacktestEngine(config)
        strategy = TestStrategyImpl()

        result = engine.run_backtest(strategy, data)

        # Verify basic result structure
        assert result is not None
        assert "metrics" in result
        metrics = result["metrics"]

        # Verify metrics are reasonable
        assert -1 <= metrics.get("total_return", 0) <= 2  # Reasonable return range
        assert metrics.get("max_drawdown", 0) <= 0  # Drawdown is negative

        # If equity is returned, verify structure
        if "equity" in result:
            equity = result["equity"]
            assert isinstance(equity, pd.Series)
            assert len(equity) > 0

    def test_stress_test_large_dataset(self):
        """Stress test with large dataset."""
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")  # ~3 years

        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        large_data = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1000000] * len(dates),
            },
            index=dates,
        )

        config = BacktestConfig(
            start_date=datetime(2020, 1, 1), end_date=datetime(2022, 12, 31), initial_capital=100000
        )

        engine = BacktestEngine(config)
        strategy = TestStrategyImpl()

        # Should handle large dataset efficiently
        result = engine.run_backtest(strategy, large_data)

        assert result is not None
        assert "metrics" in result
