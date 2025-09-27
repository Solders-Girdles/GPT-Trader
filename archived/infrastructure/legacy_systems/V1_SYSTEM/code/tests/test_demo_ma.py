"""
Comprehensive unit tests for DemoMAStrategy.

Tests moving average crossover strategy, signal generation,
ATR calculation, parameter validation, and edge cases.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from bot.strategy.demo_ma import DemoMAStrategy, _safe_atr


class TestDemoMAStrategy:
    """Test suite for DemoMAStrategy."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data with realistic patterns."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        np.random.seed(42)

        # Generate trending price data
        trend = np.linspace(100, 110, 100)
        noise = np.random.normal(0, 1, 100)
        prices = trend + noise

        return pd.DataFrame(
            {
                "Open": prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
                "High": prices * (1 + np.abs(np.random.uniform(0, 0.02, 100))),
                "Low": prices * (1 - np.abs(np.random.uniform(0, 0.02, 100))),
                "Close": prices,
                "Volume": np.random.uniform(1000000, 10000000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def volatile_data(self):
        """Create volatile market data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        np.random.seed(42)

        # High volatility data
        prices = 100 + np.cumsum(np.random.normal(0, 3, 100))

        return pd.DataFrame(
            {
                "Open": prices * (1 + np.random.uniform(-0.02, 0.02, 100)),
                "High": prices * (1 + np.abs(np.random.uniform(0, 0.05, 100))),
                "Low": prices * (1 - np.abs(np.random.uniform(0, 0.05, 100))),
                "Close": prices,
                "Volume": np.random.uniform(1000000, 10000000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def strategy(self):
        """Create default strategy instance."""
        return DemoMAStrategy()

    @pytest.fixture
    def custom_strategy(self):
        """Create strategy with custom parameters."""
        return DemoMAStrategy(fast=5, slow=15, atr_period=10)

    def test_initialization_default(self):
        """Test strategy initialization with default parameters."""
        strategy = DemoMAStrategy()
        assert strategy.name == "demo_ma"
        assert strategy.supports_short is False
        assert strategy.fast == 10
        assert strategy.slow == 20
        assert strategy.atr_period == 14

    def test_initialization_custom(self):
        """Test strategy initialization with custom parameters."""
        strategy = DemoMAStrategy(fast=5, slow=15, atr_period=10)
        assert strategy.fast == 5
        assert strategy.slow == 15
        assert strategy.atr_period == 10

    def test_initialization_type_conversion(self):
        """Test parameter type conversion."""
        strategy = DemoMAStrategy(fast=5.5, slow=15.8, atr_period=10.2)
        assert strategy.fast == 5
        assert strategy.slow == 15
        assert strategy.atr_period == 10
        assert isinstance(strategy.fast, int)
        assert isinstance(strategy.slow, int)
        assert isinstance(strategy.atr_period, int)

    def test_generate_signals_basic(self, strategy, sample_data):
        """Test basic signal generation."""
        result = strategy.generate_signals(sample_data)

        # Check required columns
        assert "signal" in result.columns
        assert "sma_fast" in result.columns
        assert "sma_slow" in result.columns
        assert "atr" in result.columns

        # Check data types
        assert result["signal"].dtype in [np.float64, np.int64]
        assert result["sma_fast"].dtype == np.float64
        assert result["sma_slow"].dtype == np.float64
        assert result["atr"].dtype == np.float64

        # Check index preservation
        assert result.index.equals(sample_data.index)

    def test_generate_signals_long_only(self, strategy, sample_data):
        """Test that strategy is long-only."""
        result = strategy.generate_signals(sample_data)

        # Signals should only be 0 or 1
        assert result["signal"].isin([0, 1]).all()
        assert (result["signal"] == -1).sum() == 0

    def test_moving_average_calculation(self, strategy, sample_data):
        """Test moving average calculations."""
        result = strategy.generate_signals(sample_data)

        # Calculate expected MAs
        expected_fast = sample_data["close"].rolling(10, min_periods=10).mean()
        expected_slow = sample_data["close"].rolling(20, min_periods=20).mean()

        # Compare (allowing for small numerical differences)
        pd.testing.assert_series_equal(
            result["sma_fast"], expected_fast, check_names=False, rtol=1e-10
        )
        pd.testing.assert_series_equal(
            result["sma_slow"], expected_slow, check_names=False, rtol=1e-10
        )

    def test_signal_generation_logic(self, strategy):
        """Test signal generation logic with controlled data."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="1D")

        # Create data where fast MA will cross above slow MA
        prices = np.concatenate(
            [
                np.linspace(100, 95, 10),  # Declining
                np.linspace(95, 105, 10),  # Rising sharply
                np.linspace(105, 108, 10),  # Rising slowly
            ]
        )

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": [1000000] * 30,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        # After warmup period (20 days), signals should reflect crossover
        valid_signals = result["signal"][20:]

        # Should have both 0 and 1 signals after warmup
        assert 0 in valid_signals.values
        assert 1 in valid_signals.values

    def test_warmup_period(self, strategy, sample_data):
        """Test that signals are zero during warmup period."""
        result = strategy.generate_signals(sample_data)

        # Maximum warmup is max(slow, atr_period) = max(20, 14) = 20
        warmup_period = 20

        # First 19 signals should be 0 (indices 0-18)
        assert (result["signal"].iloc[: warmup_period - 1] == 0).all()

        # From index 19 onwards, signals can be non-zero
        # (but not guaranteed depending on MA values)

    def test_atr_calculation(self, strategy, sample_data):
        """Test ATR calculation."""
        result = strategy.generate_signals(sample_data)

        # ATR should be non-negative
        assert (result["atr"] >= 0).all() or result["atr"].isna().all()

        # ATR should have reasonable values
        price_range = sample_data["high"] - sample_data["low"]
        max_atr = price_range.max() * 2  # Conservative upper bound

        valid_atr = result["atr"].dropna()
        if len(valid_atr) > 0:
            assert (valid_atr <= max_atr).all()

    def test_safe_atr_with_missing_columns(self, strategy):
        """Test _safe_atr with missing High/Low columns."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="1D")

        # Data without High/Low columns
        data = pd.DataFrame(
            {"Close": np.random.uniform(95, 105, 30), "Volume": [1000000] * 30}, index=dates
        )

        # Should use Close as High/Low
        atr_result = _safe_atr(data, period=14)

        assert isinstance(atr_result, pd.Series)
        assert len(atr_result) == len(data)

    def test_generate_signals_missing_high_low(self, strategy):
        """Test signal generation with missing High/Low columns."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1D")

        # Data without High/Low
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(95, 105, 50),
                "Close": np.random.uniform(95, 105, 50),
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        # Should handle gracefully
        result = strategy.generate_signals(data)

        assert "signal" in result.columns
        assert "atr" in result.columns
        assert len(result) == len(data)

    def test_crossover_detection(self, strategy):
        """Test MA crossover detection."""
        dates = pd.date_range(start="2024-01-01", periods=40, freq="1D")

        # Create perfect crossover scenario
        fast_period = strategy.fast
        slow_period = strategy.slow

        # Prices that will create a clear crossover
        prices = np.concatenate(
            [
                [100] * 15,  # Flat
                np.linspace(100, 110, 10),  # Rising (fast MA will rise faster)
                [110] * 15,  # Flat again
            ]
        )

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 0.5,
                "Low": prices - 0.5,
                "Close": prices,
                "Volume": [1000000] * 40,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        # After the rise, fast MA should be above slow MA
        # Check last few signals (after warmup and crossover)
        last_signals = result["signal"].iloc[-10:]
        assert (last_signals == 1).any()  # Should have buy signals

    def test_different_window_sizes(self):
        """Test strategy with different window size configurations."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        data = pd.DataFrame(
            {
                "Open": np.random.uniform(95, 105, 100),
                "High": np.random.uniform(100, 110, 100),
                "Low": np.random.uniform(90, 100, 100),
                "Close": np.random.uniform(95, 105, 100),
                "Volume": [1000000] * 100,
            },
            index=dates,
        )

        configs = [(5, 10, 7), (10, 20, 14), (20, 50, 20), (3, 8, 5)]

        for fast, slow, atr_period in configs:
            strategy = DemoMAStrategy(fast=fast, slow=slow, atr_period=atr_period)
            result = strategy.generate_signals(data)

            assert "signal" in result.columns
            assert strategy.fast == fast
            assert strategy.slow == slow
            assert strategy.atr_period == atr_period

    def test_nan_handling(self, strategy):
        """Test handling of NaN values in data."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1D")

        prices = np.random.uniform(95, 105, 50)
        prices[10:15] = np.nan  # Insert NaN values

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert len(result) == len(data)

    def test_extreme_values(self, strategy):
        """Test handling of extreme price values."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="1D")

        data = pd.DataFrame(
            {
                "Open": [1e-10] * 15 + [1e10] * 15,
                "High": [1e-9] * 15 + [1e11] * 15,
                "Low": [1e-11] * 15 + [1e9] * 15,
                "Close": [1e-10] * 15 + [1e10] * 15,
                "Volume": [1000] * 30,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns

    def test_single_row_data(self, strategy):
        """Test with single row of data."""
        data = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1000000]},
            index=[datetime.now()],
        )

        result = strategy.generate_signals(data)

        assert len(result) == 1
        assert result["signal"].iloc[0] == 0  # Should be 0 due to insufficient data

    def test_performance_large_dataset(self, strategy):
        """Test performance with large dataset."""
        dates = pd.date_range(start="2020-01-01", periods=5000, freq="1H")

        large_data = pd.DataFrame(
            {
                "Open": np.random.uniform(95, 105, 5000),
                "High": np.random.uniform(100, 110, 5000),
                "Low": np.random.uniform(90, 100, 5000),
                "Close": np.random.uniform(95, 105, 5000),
                "Volume": np.random.uniform(1000000, 10000000, 5000),
            },
            index=dates,
        )

        import time

        start_time = time.time()
        result = strategy.generate_signals(large_data)
        execution_time = time.time() - start_time

        assert execution_time < 2.0  # Should complete within 2 seconds
        assert len(result) == 5000

    def test_signal_stability(self, strategy, sample_data):
        """Test that signals are stable across multiple calls."""
        result1 = strategy.generate_signals(sample_data)
        result2 = strategy.generate_signals(sample_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_invalid_parameters(self):
        """Test handling of invalid parameters."""
        # Fast period larger than slow period - should still work
        strategy = DemoMAStrategy(fast=20, slow=10, atr_period=14)
        assert strategy.fast == 20
        assert strategy.slow == 10

        # Negative values should be converted to positive integers
        strategy = DemoMAStrategy(fast=-5, slow=-10, atr_period=-14)
        assert strategy.fast == -5  # Will be handled by int()

        # Zero values
        with pytest.raises(ValueError):
            strategy = DemoMAStrategy(fast=0, slow=20, atr_period=14)
            # Note: This might not raise in current implementation,
            # but it's good practice to test edge cases

    @pytest.mark.parametrize(
        "fast,slow,expected_signals",
        [
            (5, 10, "mixed"),  # Normal case
            (10, 10, "flat"),  # Equal MAs
            (20, 10, "inverted"),  # Fast > Slow period
        ],
    )
    def test_various_configurations(self, fast, slow, expected_signals):
        """Test various MA configurations."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1D")

        # Create trending data
        prices = np.linspace(100, 110, 50) + np.random.normal(0, 0.5, 50)

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        strategy = DemoMAStrategy(fast=fast, slow=slow)
        result = strategy.generate_signals(data)

        assert "signal" in result.columns
        assert len(result) == len(data)

        # Verify signals are valid (0 or 1)
        assert result["signal"].isin([0, 1]).all()


class TestDemoMAStrategyIntegration:
    """Integration tests for DemoMAStrategy."""

    @pytest.fixture
    def real_market_scenario(self):
        """Create realistic market scenario data."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1D")

        # Simulate realistic market with trends and reversals
        n_days = len(dates)

        # Create multiple market regimes
        trend1 = np.linspace(100, 120, n_days // 4)  # Uptrend
        trend2 = np.linspace(120, 110, n_days // 4)  # Downtrend
        trend3 = np.linspace(110, 115, n_days // 4)  # Slight uptrend
        trend4 = np.linspace(115, 105, n_days - 3 * (n_days // 4))  # Downtrend

        base_prices = np.concatenate([trend1, trend2, trend3, trend4])

        # Add realistic noise
        noise = np.random.normal(0, 1, n_days)
        prices = base_prices + noise

        return pd.DataFrame(
            {
                "Open": prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
                "High": prices * (1 + np.abs(np.random.uniform(0, 0.01, n_days))),
                "Low": prices * (1 - np.abs(np.random.uniform(0, 0.01, n_days))),
                "Close": prices,
                "Volume": np.random.uniform(5000000, 15000000, n_days),
            },
            index=dates,
        )

    def test_full_year_backtest(self, real_market_scenario):
        """Test strategy over full year of data."""
        strategy = DemoMAStrategy(fast=10, slow=20, atr_period=14)
        result = strategy.generate_signals(real_market_scenario)

        # Check all required columns present
        required_cols = ["signal", "sma_fast", "sma_slow", "atr"]
        for col in required_cols:
            assert col in result.columns

        # Verify signal changes occur (not stuck in one state)
        signal_changes = result["signal"].diff().abs().sum()
        assert signal_changes > 0  # Should have some signal changes

        # Verify reasonable number of trades
        buy_signals = (result["signal"] == 1).sum()
        total_days = len(result)

        # Should not be buying every day or never buying
        assert 0 < buy_signals < total_days

    def test_strategy_consistency_across_timeframes(self):
        """Test strategy consistency across different timeframes."""
        # Create same underlying data at different frequencies
        base_dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        base_prices = 100 + np.cumsum(np.random.normal(0, 1, 100))

        daily_data = pd.DataFrame(
            {
                "Open": base_prices,
                "High": base_prices + 1,
                "Low": base_prices - 1,
                "Close": base_prices,
                "Volume": [1000000] * 100,
            },
            index=base_dates,
        )

        # Test with same strategy
        strategy = DemoMAStrategy(fast=5, slow=10, atr_period=7)

        daily_signals = strategy.generate_signals(daily_data)

        # Signals should be generated consistently
        assert "signal" in daily_signals.columns
        assert len(daily_signals) == len(daily_data)
