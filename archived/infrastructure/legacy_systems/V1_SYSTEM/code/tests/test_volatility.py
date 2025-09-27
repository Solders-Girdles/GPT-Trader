"""
Unit tests for VolatilityStrategy.

Tests volatility breakout strategy using Bollinger Bands and ATR,
signal generation, parameter validation, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
from bot.strategy.volatility import VolatilityStrategy, VolatilityParams


class TestVolatilityStrategy:
    """Test suite for VolatilityStrategy."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data with volatility patterns."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        np.random.seed(42)

        # Generate data with volatility spikes
        base_price = 100
        volatility = np.random.normal(0, 1, 100)
        # Add some high volatility periods
        volatility[30:35] *= 3  # High volatility period
        volatility[60:65] *= 2.5  # Another volatility spike
        
        prices = base_price + np.cumsum(volatility)

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
                "high": prices * (1 + np.abs(np.random.uniform(0, 0.02, 100))),
                "low": prices * (1 - np.abs(np.random.uniform(0, 0.02, 100))),
                "close": prices,
                "volume": np.random.uniform(1000000, 10000000, 100),
            },
            index=dates,
        )

    @pytest.fixture
    def strategy(self):
        """Create default strategy instance."""
        return VolatilityStrategy()

    @pytest.fixture
    def custom_strategy(self):
        """Create strategy with custom parameters."""
        params = VolatilityParams(
            bb_period=10,
            bb_std_dev=1.5,
            atr_period=10,
            atr_threshold_multiplier=1.0,
            exit_middle_band=False
        )
        return VolatilityStrategy(params)

    def test_initialization_default(self):
        """Test strategy initialization with default parameters."""
        strategy = VolatilityStrategy()
        assert strategy.name == "volatility"
        assert strategy.supports_short is False
        assert strategy.params.bb_period == 20
        assert strategy.params.bb_std_dev == 2.0
        assert strategy.params.atr_period == 14
        assert strategy.params.atr_threshold_multiplier == 1.2
        assert strategy.params.exit_middle_band is True

    def test_initialization_custom(self):
        """Test strategy initialization with custom parameters."""
        params = VolatilityParams(
            bb_period=15,
            bb_std_dev=1.8,
            atr_period=12,
            atr_threshold_multiplier=1.5,
            exit_middle_band=False
        )
        strategy = VolatilityStrategy(params)
        assert strategy.params.bb_period == 15
        assert strategy.params.bb_std_dev == 1.8
        assert strategy.params.atr_period == 12
        assert strategy.params.atr_threshold_multiplier == 1.5
        assert strategy.params.exit_middle_band is False

    def test_generate_signals_basic(self, strategy, sample_data):
        """Test basic signal generation."""
        result = strategy.generate_signals(sample_data)

        # Check required columns
        assert "signal" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "atr" in result.columns
        assert "volatility_signal" in result.columns
        assert "bb_touch_signal" in result.columns

        # Check data types
        assert result["signal"].dtype in [np.float64, np.int64]
        assert result["bb_upper"].dtype == np.float64
        assert result["bb_middle"].dtype == np.float64
        assert result["bb_lower"].dtype == np.float64
        assert result["atr"].dtype == np.float64

        # Check index preservation
        assert result.index.equals(sample_data.index)

    def test_generate_signals_long_only(self, strategy, sample_data):
        """Test that strategy is long-only."""
        result = strategy.generate_signals(sample_data)

        # Signals should only be 0 or 1
        assert result["signal"].isin([0, 1]).all()
        assert (result["signal"] == -1).sum() == 0

    def test_bollinger_bands_calculation(self, strategy, sample_data):
        """Test Bollinger Bands calculations."""
        result = strategy.generate_signals(sample_data)

        # Drop NaN values for comparison
        valid_data = result.dropna()
        
        if not valid_data.empty:
            # Upper band should be >= middle band >= lower band
            assert (valid_data["bb_upper"] >= valid_data["bb_middle"]).all()
            assert (valid_data["bb_middle"] >= valid_data["bb_lower"]).all()

    def test_signal_generation_logic(self, strategy):
        """Test signal generation logic with controlled data."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1D")

        # Create data that will trigger volatility signals
        # Start with stable prices, then create a volatility breakout
        stable_prices = [100] * 25
        volatile_prices = [98, 95, 92, 88, 85, 82, 85, 88, 92, 95, 98, 100, 102, 105, 108, 110] + [110] * 9
        
        prices = np.array(stable_prices + volatile_prices)

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + np.random.uniform(0, 2, 50),
                "low": prices - np.random.uniform(0, 2, 50),
                "close": prices,
                "volume": [1000000] * 50,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        # Should have some volatility signals during the volatile period
        assert result["volatility_signal"].sum() > 0
        assert result["bb_touch_signal"].sum() > 0

    def test_warmup_period(self, strategy, sample_data):
        """Test that signals are zero during warmup period."""
        result = strategy.generate_signals(sample_data)

        # Maximum warmup is max(bb_period, atr_period) = max(20, 14) = 20
        warmup_period = 20

        # First signals should be 0 during warmup
        assert (result["signal"].iloc[:warmup_period-1] == 0).all()

    def test_atr_calculation(self, strategy, sample_data):
        """Test ATR calculation."""
        result = strategy.generate_signals(sample_data)

        # ATR should be non-negative
        valid_atr = result["atr"].dropna()
        if len(valid_atr) > 0:
            assert (valid_atr >= 0).all()

    def test_insufficient_data(self, strategy):
        """Test handling of insufficient data."""
        # Create data with fewer periods than required
        dates = pd.date_range(start="2024-01-01", periods=15, freq="1D")
        data = pd.DataFrame(
            {
                "open": [100] * 15,
                "high": [101] * 15,
                "low": [99] * 15,
                "close": [100] * 15,
                "volume": [1000000] * 15,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        # Should return all zeros for signals
        assert (result["signal"] == 0).all()
        assert len(result) == len(data)

    def test_empty_data(self, strategy):
        """Test handling of empty DataFrame."""
        empty_data = pd.DataFrame()
        result = strategy.generate_signals(empty_data)

        # Should return empty DataFrame with correct structure
        expected_columns = [
            "signal", "bb_upper", "bb_middle", "bb_lower", 
            "atr", "volatility_signal", "bb_touch_signal"
        ]
        for col in expected_columns:
            assert col in result.columns

    def test_custom_parameters(self, custom_strategy, sample_data):
        """Test strategy with custom parameters."""
        result = custom_strategy.generate_signals(sample_data)

        # Should still generate valid signals
        assert "signal" in result.columns
        assert result["signal"].isin([0, 1]).all()

        # Different parameters should potentially give different results
        default_strategy = VolatilityStrategy()
        default_result = default_strategy.generate_signals(sample_data)

        # Results might be different due to different parameters
        # Just check that both produce valid outputs
        assert len(result) == len(default_result)

    def test_exit_conditions(self):
        """Test different exit conditions."""
        dates = pd.date_range(start="2024-01-01", periods=60, freq="1D")
        
        # Create scenario where price hits lower band, then rises
        prices = np.concatenate([
            [100] * 25,  # Stable
            [98, 95, 92],  # Drop to lower band
            np.linspace(92, 105, 10),  # Rise through middle to upper
            [105] * 22  # Stable high
        ])

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": [1000000] * 60,
            },
            index=dates,
        )

        # Test with exit_middle_band=True
        params_middle = VolatilityParams(exit_middle_band=True)
        strategy_middle = VolatilityStrategy(params_middle)
        result_middle = strategy_middle.generate_signals(data)

        # Test with exit_middle_band=False
        params_upper = VolatilityParams(exit_middle_band=False)
        strategy_upper = VolatilityStrategy(params_upper)
        result_upper = strategy_upper.generate_signals(data)

        # Both should be valid
        assert result_middle["signal"].isin([0, 1]).all()
        assert result_upper["signal"].isin([0, 1]).all()

    def test_string_representation(self):
        """Test string representations."""
        strategy = VolatilityStrategy()
        
        str_repr = str(strategy)
        assert "VolatilityStrategy" in str_repr
        assert "bb_period=20" in str_repr
        assert "atr_period=14" in str_repr
        
        repr_str = repr(strategy)
        assert repr_str == str_repr

    def test_signal_stability(self, strategy, sample_data):
        """Test that signals are stable across multiple calls."""
        result1 = strategy.generate_signals(sample_data)
        result2 = strategy.generate_signals(sample_data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_missing_columns(self, strategy):
        """Test handling of missing high/low columns."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1D")

        # Data without high/low columns
        data = pd.DataFrame(
            {
                "open": np.random.uniform(95, 105, 50),
                "close": np.random.uniform(95, 105, 50),
                "volume": [1000000] * 50,
            },
            index=dates,
        )

        # Should handle gracefully using _safe_atr
        result = strategy.generate_signals(data)

        assert "signal" in result.columns
        assert "atr" in result.columns
        assert len(result) == len(data)

    def test_nan_handling(self, strategy):
        """Test handling of NaN values in data."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1D")

        prices = np.random.uniform(95, 105, 50)
        prices[10:15] = np.nan  # Insert NaN values

        data = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": [1000000] * 50,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert len(result) == len(data)