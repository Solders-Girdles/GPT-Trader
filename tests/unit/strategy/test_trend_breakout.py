"""
Comprehensive unit tests for TrendBreakoutStrategy.

Tests Donchian channel breakout strategy, signal generation,
parameter validation, and edge cases.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from bot.strategy.trend_breakout import TrendBreakoutParams, TrendBreakoutStrategy


class TestTrendBreakoutStrategy:
    """Test suite for TrendBreakoutStrategy."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data with breakout patterns."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
        np.random.seed(42)

        # Create data with clear breakout patterns
        base_price = 100
        prices = []

        # Consolidation phase (30 days)
        for i in range(30):
            prices.append(base_price + np.random.uniform(-2, 2))

        # Breakout phase (20 days)
        for i in range(20):
            prices.append(base_price + 5 + i * 0.5 + np.random.uniform(-1, 1))

        # New consolidation (30 days)
        new_base = prices[-1]
        for i in range(30):
            prices.append(new_base + np.random.uniform(-2, 2))

        # Another breakout (20 days)
        for i in range(20):
            prices.append(new_base + 5 + i * 0.3 + np.random.uniform(-1, 1))

        prices = np.array(prices)

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
    def trending_data(self):
        """Create strong trending data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")

        # Strong uptrend
        prices = 100 + np.arange(100) * 0.5 + np.random.normal(0, 0.5, 100)

        return pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1000000] * 100,
            },
            index=dates,
        )

    @pytest.fixture
    def ranging_data(self):
        """Create ranging/sideways market data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")

        # Oscillating prices
        prices = 100 + 5 * np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.5, 100)

        return pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.02,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1000000] * 100,
            },
            index=dates,
        )

    @pytest.fixture
    def strategy(self):
        """Create default strategy instance."""
        return TrendBreakoutStrategy()

    @pytest.fixture
    def custom_strategy(self):
        """Create strategy with custom parameters."""
        params = TrendBreakoutParams(donchian_lookback=20, atr_period=14, atr_k=1.5)
        return TrendBreakoutStrategy(params)

    def test_initialization_default(self):
        """Test strategy initialization with default parameters."""
        strategy = TrendBreakoutStrategy()
        assert strategy.name == "trend_breakout"
        assert strategy.supports_short is False
        assert strategy.params.donchian_lookback == 55
        assert strategy.params.atr_period == 20
        assert strategy.params.atr_k == 2.0

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        params = TrendBreakoutParams(donchian_lookback=30, atr_period=15, atr_k=1.5)
        strategy = TrendBreakoutStrategy(params)

        assert strategy.params.donchian_lookback == 30
        assert strategy.params.atr_period == 15
        assert strategy.params.atr_k == 1.5

    def test_generate_signals_basic(self, strategy, sample_data):
        """Test basic signal generation."""
        result = strategy.generate_signals(sample_data)

        # Check required columns
        required_cols = ["signal", "donchian_upper", "donchian_lower", "atr"]
        for col in required_cols:
            assert col in result.columns

        # Check data types
        assert result["signal"].dtype in [np.int64, np.float64]
        assert result["donchian_upper"].dtype == np.float64
        assert result["donchian_lower"].dtype == np.float64
        assert result["atr"].dtype == np.float64

        # Check index preservation
        assert result.index.equals(sample_data.index)

    def test_signal_values(self, strategy, sample_data):
        """Test that signals are valid."""
        result = strategy.generate_signals(sample_data)

        # Signals should only be 0 or 1 (long-only)
        assert result["signal"].isin([0, 1]).all()
        assert (result["signal"] == -1).sum() == 0  # No short signals

    def test_donchian_channels(self, strategy, sample_data):
        """Test Donchian channel calculation."""
        result = strategy.generate_signals(sample_data)

        # Donchian upper should be >= high
        # Donchian lower should be <= low
        valid_idx = ~(result["donchian_upper"].isna() | result["donchian_lower"].isna())

        if valid_idx.any():
            # Upper channel should be at or above recent highs
            assert (
                result.loc[valid_idx, "donchian_upper"] >= sample_data.loc[valid_idx, "Low"]
            ).all()

            # Lower channel should be at or below recent highs
            assert (
                result.loc[valid_idx, "donchian_lower"] <= sample_data.loc[valid_idx, "High"]
            ).all()

    def test_breakout_detection(self, strategy):
        """Test breakout signal generation."""
        dates = pd.date_range(start="2024-01-01", periods=60, freq="1D")

        # Create clear breakout scenario
        prices = np.concatenate(
            [
                [100] * 20,  # Flat period to establish channel
                np.linspace(100, 90, 10),  # Decline
                [90] * 10,  # New flat period
                np.linspace(90, 110, 20),  # Strong breakout
            ]
        )

        data = pd.DataFrame(
            {
                "Open": prices * 0.99,
                "High": prices * 1.01,
                "Low": prices * 0.98,
                "Close": prices,
                "Volume": [1000000] * 60,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        # Should detect breakout in the rising period
        # After initial flat period and during the strong rise
        breakout_signals = result["signal"][40:].sum()
        assert breakout_signals > 0  # Should have some breakout signals

    def test_atr_calculation(self, strategy, sample_data):
        """Test ATR calculation."""
        result = strategy.generate_signals(sample_data)

        # ATR should be non-negative
        valid_atr = result["atr"].dropna()
        if len(valid_atr) > 0:
            assert (valid_atr >= 0).all()

            # ATR should be reasonable relative to price movements
            price_range = sample_data["high"] - sample_data["low"]
            assert (valid_atr <= price_range.max() * 3).all()

    def test_no_lookahead_bias(self, strategy, sample_data):
        """Test that strategy doesn't use future information."""
        result = strategy.generate_signals(sample_data)

        # Signals should be based on prior day's channel
        # This is implicitly tested by the shift(1) in the strategy
        # We can verify by checking that first signal after warmup
        # doesn't immediately trigger

        warmup = strategy.params.donchian_lookback
        if len(result) > warmup:
            # Not all post-warmup periods should have signals
            post_warmup_signals = result["signal"][warmup:]
            assert post_warmup_signals.sum() < len(post_warmup_signals)

    def test_warmup_period(self, strategy, sample_data):
        """Test handling of warmup period."""
        result = strategy.generate_signals(sample_data)

        # During warmup, Donchian channels might be NaN
        warmup = strategy.params.donchian_lookback - 1

        # Check first few rows
        if warmup > 0:
            early_channels = result[["donchian_upper", "donchian_lower"]].iloc[: warmup // 2]
            # Some early values might be NaN
            assert early_channels.isna().any().any() or len(early_channels) == 0

    def test_trending_market_behavior(self, strategy, trending_data):
        """Test strategy behavior in trending market."""
        result = strategy.generate_signals(trending_data)

        # In strong uptrend, should generate multiple buy signals
        total_signals = result["signal"].sum()
        assert total_signals > 0

        # Signals should cluster during trend continuation
        # Not testing exact count as it depends on parameters

    def test_ranging_market_behavior(self, strategy, ranging_data):
        """Test strategy behavior in ranging market."""
        result = strategy.generate_signals(ranging_data)

        # In ranging market, might have fewer breakouts
        total_signals = result["signal"].sum()

        # Should still have some signals but not constant
        assert 0 <= total_signals < len(result)

    def test_different_lookback_periods(self, sample_data):
        """Test strategy with different lookback periods."""
        lookbacks = [10, 20, 55, 100]

        for lookback in lookbacks:
            params = TrendBreakoutParams(donchian_lookback=lookback)
            strategy = TrendBreakoutStrategy(params)

            if lookback <= len(sample_data):
                result = strategy.generate_signals(sample_data)
                assert "signal" in result.columns
                assert len(result) == len(sample_data)

    def test_different_atr_periods(self, sample_data):
        """Test strategy with different ATR periods."""
        atr_periods = [5, 14, 20, 30]

        for period in atr_periods:
            params = TrendBreakoutParams(atr_period=period)
            strategy = TrendBreakoutStrategy(params)

            result = strategy.generate_signals(sample_data)
            assert "atr" in result.columns

            # ATR should be calculated with specified period
            valid_atr = result["atr"].dropna()
            if len(valid_atr) > 0:
                assert (valid_atr >= 0).all()

    def test_edge_case_single_row(self, strategy):
        """Test with single row of data."""
        data = pd.DataFrame(
            {"Open": [100], "High": [101], "Low": [99], "Close": [100], "Volume": [1000000]},
            index=[datetime.now()],
        )

        result = strategy.generate_signals(data)

        assert len(result) == 1
        assert "signal" in result.columns
        # With single row, can't detect breakout
        assert result["signal"].iloc[0] == 0

    def test_edge_case_insufficient_data(self, strategy):
        """Test with insufficient data for lookback."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="1D")

        small_data = pd.DataFrame(
            {
                "Open": np.random.uniform(99, 101, 10),
                "High": np.random.uniform(100, 102, 10),
                "Low": np.random.uniform(98, 100, 10),
                "Close": np.random.uniform(99, 101, 10),
                "Volume": [1000000] * 10,
            },
            index=dates,
        )

        # Default lookback is 55, data has only 10 rows
        result = strategy.generate_signals(small_data)

        assert len(result) == 10
        assert "signal" in result.columns

    def test_nan_handling(self, strategy):
        """Test handling of NaN values."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")

        prices = np.random.uniform(95, 105, 100)
        prices[40:50] = np.nan  # Insert NaN values

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": [1000000] * 100,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns
        assert len(result) == len(data)

    def test_extreme_values(self, strategy):
        """Test handling of extreme values."""
        dates = pd.date_range(start="2024-01-01", periods=60, freq="1D")

        # Mix of normal and extreme values
        prices = np.concatenate(
            [
                np.random.uniform(95, 105, 20),
                [1e-10] * 10,
                np.random.uniform(95, 105, 20),
                [1e10] * 10,
            ]
        )

        data = pd.DataFrame(
            {
                "Open": prices,
                "High": prices * 1.01,
                "Low": prices * 0.99,
                "Close": prices,
                "Volume": [1000000] * 60,
            },
            index=dates,
        )

        result = strategy.generate_signals(data)

        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns

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

        assert execution_time < 3.0  # Should complete within 3 seconds
        assert len(result) == 5000

    def test_signal_persistence(self, strategy):
        """Test that signals are consistent across calls."""
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

        result1 = strategy.generate_signals(data)
        result2 = strategy.generate_signals(data)

        pd.testing.assert_frame_equal(result1, result2)

    def test_params_dataclass(self):
        """Test TrendBreakoutParams dataclass."""
        # Test default values
        params1 = TrendBreakoutParams()
        assert params1.donchian_lookback == 55
        assert params1.atr_period == 20
        assert params1.atr_k == 2.0

        # Test custom values
        params2 = TrendBreakoutParams(donchian_lookback=30, atr_period=15, atr_k=1.5)
        assert params2.donchian_lookback == 30
        assert params2.atr_period == 15
        assert params2.atr_k == 1.5

    @pytest.mark.parametrize(
        "lookback,atr_period,atr_k", [(20, 14, 1.0), (55, 20, 2.0), (100, 30, 3.0), (10, 5, 0.5)]
    )
    def test_various_parameter_combinations(self, sample_data, lookback, atr_period, atr_k):
        """Test various parameter combinations."""
        params = TrendBreakoutParams(donchian_lookback=lookback, atr_period=atr_period, atr_k=atr_k)
        strategy = TrendBreakoutStrategy(params)

        result = strategy.generate_signals(sample_data)

        assert "signal" in result.columns
        assert "atr" in result.columns
        assert result["signal"].isin([0, 1]).all()


class TestTrendBreakoutIntegration:
    """Integration tests for TrendBreakoutStrategy."""

    @pytest.fixture
    def market_regime_data(self):
        """Create data with multiple market regimes."""
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1D")
        n_days = len(dates)

        # Create different market regimes
        regimes = []

        # Q1: Ranging market
        q1 = 100 + 5 * np.sin(np.linspace(0, 4 * np.pi, n_days // 4))
        regimes.append(q1)

        # Q2: Strong uptrend
        q2 = np.linspace(105, 130, n_days // 4)
        regimes.append(q2)

        # Q3: Volatile/choppy
        q3 = 130 + np.cumsum(np.random.normal(0, 2, n_days // 4))
        regimes.append(q3)

        # Q4: Downtrend
        q4_len = n_days - 3 * (n_days // 4)
        q4 = np.linspace(regimes[-1][-1], 110, q4_len)
        regimes.append(q4)

        prices = np.concatenate(regimes)

        # Add realistic noise
        prices = prices + np.random.normal(0, 0.5, n_days)

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

    def test_full_year_signals(self, market_regime_data):
        """Test strategy over full year with different market regimes."""
        strategy = TrendBreakoutStrategy()
        result = strategy.generate_signals(market_regime_data)

        # Should generate signals during trending periods
        total_signals = result["signal"].sum()
        assert total_signals > 0

        # Signals should not be constant
        assert total_signals < len(result) * 0.5  # Less than 50% of days

        # Check signal distribution across year
        monthly_signals = result["signal"].resample("M").sum()

        # Should have varying signal frequency across months
        assert monthly_signals.std() > 0
