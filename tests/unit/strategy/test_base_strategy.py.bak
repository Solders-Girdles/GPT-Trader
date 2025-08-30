"""
Unit tests for base Strategy class.

Tests the abstract base class and common strategy functionality.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from bot.strategy.base import Strategy


class ConcreteStrategy(Strategy):
    """Concrete implementation for testing."""

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate simple test signals."""
        return pd.Series([1, -1, 0] * (len(data) // 3 + 1))[: len(data)]

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate test indicators."""
        data["test_indicator"] = data["Close"].rolling(window=5).mean()
        return data


class TestStrategy:
    """Test suite for Strategy base class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        return pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 100),
                "High": np.random.uniform(110, 120, 100),
                "Low": np.random.uniform(90, 100, 100),
                "Close": np.random.uniform(95, 115, 100),
                "Volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def strategy(self):
        """Create concrete strategy instance."""
        return ConcreteStrategy(name="test_strategy", lookback_period=20, confidence_threshold=0.6)

    def test_strategy_initialization(self):
        """Test strategy initialization with parameters."""
        strategy = ConcreteStrategy(name="test", lookback_period=30, confidence_threshold=0.7)
        assert strategy.name == "test"
        assert strategy.lookback_period == 30
        assert strategy.confidence_threshold == 0.7

    def test_strategy_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            Strategy(name="abstract")

    def test_generate_signals(self, strategy, sample_data):
        """Test signal generation."""
        signals = strategy.generate_signals(sample_data)

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert all(signal in [-1, 0, 1] for signal in signals.unique())

    def test_calculate_indicators(self, strategy, sample_data):
        """Test indicator calculation."""
        data_with_indicators = strategy.calculate_indicators(sample_data.copy())

        assert "test_indicator" in data_with_indicators.columns
        assert len(data_with_indicators) == len(sample_data)

    def test_validate_data(self, strategy, sample_data):
        """Test data validation."""
        # Should not raise for valid data
        strategy.validate_data(sample_data)

        # Test with missing columns
        invalid_data = sample_data.drop(columns=["Close"])
        with pytest.raises(ValueError, match="Missing required columns"):
            strategy.validate_data(invalid_data)

        # Test with empty data
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="Empty data"):
            strategy.validate_data(empty_data)

    def test_backtest(self, strategy, sample_data):
        """Test backtesting functionality."""
        results = strategy.backtest(sample_data)

        assert "returns" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "win_rate" in results
        assert isinstance(results["returns"], (float, np.floating))

    def test_optimize_parameters(self, strategy, sample_data):
        """Test parameter optimization."""
        param_ranges = {
            "lookback_period": [10, 20, 30],
            "confidence_threshold": [0.5, 0.6, 0.7],
        }

        best_params = strategy.optimize_parameters(sample_data, param_ranges)

        assert "lookback_period" in best_params
        assert "confidence_threshold" in best_params
        assert best_params["lookback_period"] in param_ranges["lookback_period"]

    def test_get_performance_metrics(self, strategy, sample_data):
        """Test performance metrics calculation."""
        signals = strategy.generate_signals(sample_data)
        metrics = strategy.get_performance_metrics(sample_data, signals)

        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())

    @pytest.mark.parametrize(
        "lookback,threshold",
        [(10, 0.5), (20, 0.6), (30, 0.7), (50, 0.8)],
    )
    def test_strategy_with_different_parameters(self, sample_data, lookback, threshold):
        """Test strategy with various parameter combinations."""
        strategy = ConcreteStrategy(
            name="param_test",
            lookback_period=lookback,
            confidence_threshold=threshold,
        )

        signals = strategy.generate_signals(sample_data)
        assert len(signals) == len(sample_data)

    def test_strategy_persistence(self, strategy, tmp_path):
        """Test saving and loading strategy state."""
        filepath = tmp_path / "strategy_state.joblib"

        # Save strategy
        strategy.save_state(filepath)
        assert filepath.exists()

        # Load strategy
        loaded_strategy = ConcreteStrategy.load_state(filepath)
        assert loaded_strategy.name == strategy.name
        assert loaded_strategy.lookback_period == strategy.lookback_period

    def test_strategy_with_live_data(self, strategy):
        """Test strategy with live/streaming data."""
        # Simulate live data point
        live_data = pd.DataFrame(
            {
                "Open": [105.0],
                "High": [110.0],
                "Low": [104.0],
                "Close": [108.0],
                "Volume": [5000],
            },
            index=[datetime.now()],
        )

        signal = strategy.process_live_data(live_data)
        assert signal in [-1, 0, 1]

    def test_strategy_error_handling(self, strategy):
        """Test strategy error handling."""
        # Test with invalid data types
        with pytest.raises(TypeError):
            strategy.generate_signals("not_a_dataframe")

        # Test with NaN values
        data_with_nan = pd.DataFrame(
            {"Close": [100, np.nan, 102, 103, 104]},
            index=pd.date_range("2024-01-01", periods=5),
        )

        with pytest.raises(ValueError, match="contains NaN"):
            strategy.validate_data(data_with_nan)

    def test_strategy_thread_safety(self, strategy, sample_data):
        """Test strategy thread safety for concurrent operations."""
        import threading

        results = []

        def run_strategy():
            signals = strategy.generate_signals(sample_data)
            results.append(len(signals))

        threads = [threading.Thread(target=run_strategy) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == len(sample_data) for r in results)

    @pytest.mark.slow
    def test_strategy_performance(self, strategy):
        """Test strategy performance with large dataset."""
        large_data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 110, 10000),
                "High": np.random.uniform(110, 120, 10000),
                "Low": np.random.uniform(90, 100, 10000),
                "Close": np.random.uniform(95, 115, 10000),
                "Volume": np.random.uniform(1000, 10000, 10000),
            },
            index=pd.date_range("2020-01-01", periods=10000, freq="1h"),
        )

        import time

        start = time.time()
        signals = strategy.generate_signals(large_data)
        execution_time = time.time() - start

        assert len(signals) == len(large_data)
        assert execution_time < 5.0  # Should complete within 5 seconds
