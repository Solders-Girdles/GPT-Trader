"""
Comprehensive unit tests for base Strategy class.

Tests abstract base class, signal generation, indicator calculation,
error handling, and edge cases.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest
from bot.strategy.base import Strategy


class ConcreteStrategy(Strategy):
    """Concrete implementation for testing."""

    name = "test_strategy"
    supports_short = False

    def __init__(self, config: dict[str, Any] = None):
        self.config = config or {}
        self.call_count = 0

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate test signals based on simple MA crossover."""
        self.call_count += 1

        # Calculate indicators
        bars = bars.copy()
        bars["sma_fast"] = bars["Close"].rolling(window=5, min_periods=1).mean()
        bars["sma_slow"] = bars["Close"].rolling(window=20, min_periods=1).mean()

        # Generate signals
        bars["signal"] = 0
        bars.loc[bars["sma_fast"] > bars["sma_slow"], "signal"] = 1
        bars.loc[bars["sma_fast"] < bars["sma_slow"], "signal"] = 0

        # Add additional columns for testing
        bars["confidence"] = np.random.uniform(0.5, 1.0, len(bars))
        bars["atr"] = bars["Close"].rolling(window=14, min_periods=1).std()

        return bars


class ShortSellingStrategy(Strategy):
    """Strategy that supports short selling."""

    name = "short_strategy"
    supports_short = True

    def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Generate signals including short positions."""
        bars = bars.copy()
        bars["rsi"] = 50 + np.random.uniform(-30, 30, len(bars))

        bars["signal"] = 0
        bars.loc[bars["rsi"] > 70, "signal"] = -1  # Short
        bars.loc[bars["rsi"] < 30, "signal"] = 1  # Long

        return bars


class TestStrategyBase:
    """Test suite for Strategy base class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.0001, 0.01, 100)
        prices = 100 * np.exp(np.cumsum(returns))

        return pd.DataFrame(
            {
                "Open": prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
                "High": prices * (1 + np.abs(np.random.uniform(0, 0.01, 100))),
                "Low": prices * (1 - np.abs(np.random.uniform(0, 0.01, 100))),
                "Close": prices,
                "Volume": np.random.uniform(1000000, 10000000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def strategy(self):
        """Create concrete strategy instance."""
        return ConcreteStrategy()

    @pytest.fixture
    def short_strategy(self):
        """Create short-selling strategy instance."""
        return ShortSellingStrategy()

    def test_strategy_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.name == "test_strategy"
        assert strategy.supports_short is False
        assert strategy.config == {}
        assert strategy.call_count == 0

    def test_strategy_with_config(self):
        """Test strategy initialization with config."""
        config = {"param1": 10, "param2": "value"}
        strategy = ConcreteStrategy(config=config)
        assert strategy.config == config

    def test_generate_signals_basic(self, strategy, sample_data):
        """Test basic signal generation."""
        result = strategy.generate_signals(sample_data)

        # Check required columns
        assert "signal" in result.columns
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "confidence" in result.columns
        assert "atr" in result.columns

        # Check signal values
        assert result["signal"].isin([0, 1]).all()

        # Check data integrity
        assert len(result) == len(sample_data)
        assert result.index.equals(sample_data.index)

    def test_generate_signals_long_only(self, strategy, sample_data):
        """Test that long-only strategy doesn't generate short signals."""
        result = strategy.generate_signals(sample_data)

        # Long-only strategy should only have 0 or 1 signals
        assert result["signal"].isin([0, 1]).all()
        assert (result["signal"] == -1).sum() == 0

    def test_generate_signals_short_selling(self, short_strategy, sample_data):
        """Test short-selling strategy generates all signal types."""
        result = short_strategy.generate_signals(sample_data)

        # Should have long, flat, and short signals
        assert result["signal"].isin([-1, 0, 1]).all()

        # Verify short signals exist
        assert (result["signal"] == -1).sum() > 0
        assert (result["signal"] == 1).sum() > 0

    def test_generate_signals_empty_data(self, strategy):
        """Test signal generation with empty data."""
        empty_data = pd.DataFrame()
        result = strategy.generate_signals(empty_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_generate_signals_single_row(self, strategy):
        """Test signal generation with single row of data."""
        single_row = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000000],
            },
            index=[datetime.now()],
        )

        result = strategy.generate_signals(single_row)

        assert len(result) == 1
        assert "signal" in result.columns

    def test_generate_signals_missing_columns(self, strategy):
        """Test signal generation with missing required columns."""
        incomplete_data = pd.DataFrame({"Close": [100, 101, 102]})

        # Should handle gracefully or raise appropriate error
        with pytest.raises(KeyError):
            strategy.generate_signals(incomplete_data)

    def test_generate_signals_nan_values(self, strategy, sample_data):
        """Test signal generation with NaN values."""
        sample_data.loc[sample_data.index[10:20], "Close"] = np.nan

        result = strategy.generate_signals(sample_data)

        # Strategy should handle NaN values appropriately
        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns

    def test_generate_signals_extreme_values(self, strategy):
        """Test signal generation with extreme price values."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1h")

        extreme_data = pd.DataFrame(
            {
                "Open": [1e-10] * 25 + [1e10] * 25,
                "High": [1e-9] * 25 + [1e11] * 25,
                "Low": [1e-11] * 25 + [1e9] * 25,
                "Close": [1e-10] * 25 + [1e10] * 25,
                "Volume": [1000] * 50,
            },
            index=dates,
        )

        result = strategy.generate_signals(extreme_data)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns

    def test_generate_signals_call_count(self, strategy, sample_data):
        """Test that generate_signals increments call count."""
        assert strategy.call_count == 0

        strategy.generate_signals(sample_data)
        assert strategy.call_count == 1

        strategy.generate_signals(sample_data)
        assert strategy.call_count == 2

    def test_generate_signals_data_immutability(self, strategy, sample_data):
        """Test that original data is not modified."""
        original_data = sample_data.copy()

        result = strategy.generate_signals(sample_data)

        # Original data should not be modified
        pd.testing.assert_frame_equal(sample_data, original_data)

        # Result should be a different object
        assert result is not sample_data

    def test_abstract_base_class(self):
        """Test that Strategy is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            Strategy()

    def test_indicator_calculation(self, strategy, sample_data):
        """Test that indicators are calculated correctly."""
        result = strategy.generate_signals(sample_data)

        # Verify SMA calculations
        expected_sma_fast = sample_data["Close"].rolling(window=5, min_periods=1).mean()
        pd.testing.assert_series_equal(result["sma_fast"], expected_sma_fast, check_names=False)

        expected_sma_slow = sample_data["Close"].rolling(window=20, min_periods=1).mean()
        pd.testing.assert_series_equal(result["sma_slow"], expected_sma_slow, check_names=False)

    def test_signal_consistency(self, strategy, sample_data):
        """Test that signals are consistent across multiple calls."""
        np.random.seed(42)
        result1 = strategy.generate_signals(sample_data)

        np.random.seed(42)
        result2 = strategy.generate_signals(sample_data)

        # Signals should be deterministic given same input
        pd.testing.assert_series_equal(result1["signal"], result2["signal"])

    def test_signal_transitions(self, strategy, sample_data):
        """Test signal transitions are logical."""
        result = strategy.generate_signals(sample_data)

        # Calculate signal changes
        signal_diff = result["signal"].diff()

        # Transitions should be reasonable (not jumping from 1 to -1 for long-only)
        if not strategy.supports_short:
            # For long-only, transitions should only be 0->1, 1->0
            assert signal_diff.isin([0, 1, -1, np.nan]).all()

    def test_performance_with_large_dataset(self, strategy):
        """Test strategy performance with large dataset."""
        # Create large dataset
        dates = pd.date_range(start="2020-01-01", periods=10000, freq="1min")
        large_data = pd.DataFrame(
            {
                "Open": np.random.uniform(99, 101, 10000),
                "High": np.random.uniform(100, 102, 10000),
                "Low": np.random.uniform(98, 100, 10000),
                "Close": np.random.uniform(99, 101, 10000),
                "Volume": np.random.uniform(1000, 10000, 10000),
            },
            index=dates,
        )

        import time

        start_time = time.time()
        result = strategy.generate_signals(large_data)
        execution_time = time.time() - start_time

        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 seconds max
        assert len(result) == 10000

    @pytest.mark.parametrize("window_size", [5, 10, 20, 50])
    def test_different_window_sizes(self, sample_data, window_size):
        """Test strategy with different window sizes."""

        class ParameterizedStrategy(Strategy):
            name = "param_strategy"
            supports_short = False

            def generate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
                bars = bars.copy()
                bars["sma"] = bars["Close"].rolling(window=window_size, min_periods=1).mean()
                bars["signal"] = (bars["Close"] > bars["sma"]).astype(int)
                return bars

        strategy = ParameterizedStrategy()
        result = strategy.generate_signals(sample_data)

        assert "signal" in result.columns
        assert result["signal"].isin([0, 1]).all()


class TestStrategyEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def strategy(self):
        return ConcreteStrategy()

    def test_weekend_gaps(self, strategy):
        """Test handling of weekend gaps in data."""
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="B")  # Business days only

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(99, 101, len(dates)),
                "High": np.random.uniform(100, 102, len(dates)),
                "Low": np.random.uniform(98, 100, len(dates)),
                "Close": np.random.uniform(99, 101, len(dates)),
                "Volume": np.random.uniform(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        assert len(result) == len(data)
        assert "signal" in result.columns

    def test_duplicate_timestamps(self, strategy):
        """Test handling of duplicate timestamps."""
        dates = [datetime.now()] * 5

        data = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000000] * 5,
            },
            index=dates,
        )

        # Should handle duplicate timestamps appropriately
        result = strategy.generate_signals(data)
        assert isinstance(result, pd.DataFrame)

    def test_non_chronological_data(self, strategy):
        """Test handling of non-chronological data."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="1h")
        shuffled_dates = dates.to_list()
        np.random.shuffle(shuffled_dates)

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(99, 101, 10),
                "High": np.random.uniform(100, 102, 10),
                "Low": np.random.uniform(98, 100, 10),
                "Close": np.random.uniform(99, 101, 10),
                "Volume": np.random.uniform(1000000, 10000000, 10),
            },
            index=shuffled_dates,
        )

        result = strategy.generate_signals(data)
        assert len(result) == len(data)

    def test_zero_volume(self, strategy):
        """Test handling of zero volume periods."""
        dates = pd.date_range(start="2024-01-01", periods=20, freq="1h")

        data = pd.DataFrame(
            {
                "Open": np.random.uniform(99, 101, 20),
                "High": np.random.uniform(100, 102, 20),
                "Low": np.random.uniform(98, 100, 20),
                "Close": np.random.uniform(99, 101, 20),
                "Volume": [0] * 10 + [1000000] * 10,  # First 10 periods have zero volume
            },
            index=dates,
        )

        result = strategy.generate_signals(data)
        assert "signal" in result.columns

    def test_negative_prices(self, strategy):
        """Test handling of negative prices (should not occur but test robustness)."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="1h")

        data = pd.DataFrame(
            {
                "Open": [-100, -50, 0, 50, 100, 100, 100, 100, 100, 100],
                "High": [-90, -40, 10, 60, 110, 110, 110, 110, 110, 110],
                "Low": [-110, -60, -10, 40, 90, 90, 90, 90, 90, 90],
                "Close": [-95, -45, 5, 55, 105, 105, 105, 105, 105, 105],
                "Volume": [1000000] * 10,
            },
            index=dates,
        )

        # Strategy should handle or reject negative prices appropriately
        result = strategy.generate_signals(data)
        assert isinstance(result, pd.DataFrame)
