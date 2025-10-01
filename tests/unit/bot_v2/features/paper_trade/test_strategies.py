"""
Comprehensive tests for paper trading strategies.

Tests cover:
- Strategy initialization and parameter validation
- Signal generation for all strategy types
- Required periods calculation
- Strategy factory creation
- Edge cases and boundary conditions
- Position tracking
"""

import pandas as pd
import pytest

from bot_v2.features.paper_trade.strategies import (
    BreakoutStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    SimpleMAStrategy,
    VolatilityStrategy,
    create_paper_strategy,
)


# ============================================================================
# Helper Functions
# ============================================================================


def create_sample_data(
    num_rows: int, base_price: float = 100.0, trend: str = "flat", volatility: float = 0.02
) -> pd.DataFrame:
    """
    Create sample market data for testing.

    Args:
        num_rows: Number of data rows
        base_price: Starting price
        trend: Price trend ('up', 'down', 'flat', 'oscillating')
        volatility: Price volatility as percentage

    Returns:
        DataFrame with OHLCV data
    """
    import numpy as np

    np.random.seed(42)

    prices = [base_price]
    for i in range(1, num_rows):
        if trend == "up":
            change = 1 + (0.001 + np.random.randn() * volatility)
        elif trend == "down":
            change = 1 - (0.001 + np.random.randn() * volatility)
        elif trend == "oscillating":
            change = 1 + np.sin(i / 5) * 0.01 + np.random.randn() * volatility
        else:  # flat
            change = 1 + np.random.randn() * volatility

        prices.append(prices[-1] * change)

    prices = pd.Series(prices)
    high = prices * (1 + abs(np.random.randn(num_rows) * volatility / 2))
    low = prices * (1 - abs(np.random.randn(num_rows) * volatility / 2))
    volume = np.random.randint(1000000, 5000000, num_rows)

    return pd.DataFrame(
        {"open": prices, "high": high, "low": low, "close": prices, "volume": volume}
    )


# ============================================================================
# Test: SimpleMAStrategy
# ============================================================================


class TestSimpleMAStrategy:
    """Test Simple Moving Average strategy."""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters."""
        strategy = SimpleMAStrategy()

        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.position == 0
        assert "fast_period" in strategy.params
        assert "slow_period" in strategy.params

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = SimpleMAStrategy(fast_period=5, slow_period=20)

        assert strategy.fast_period == 5
        assert strategy.slow_period == 20

    def test_get_required_periods(self):
        """Test required periods calculation."""
        strategy = SimpleMAStrategy(fast_period=10, slow_period=30)

        assert strategy.get_required_periods() == 31

    def test_bullish_crossover_signal(self):
        """Test detection of bullish MA crossover (fast crosses above slow)."""
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)

        # Create controlled data where crossover happens at the very end
        # Pattern: Flat, then decline, then sharp reversal
        prices = [100.0] * 36  # Flat for 36 days

        # Days 36-40: Decline (fast MA will drop below slow MA)
        prices.extend([98.0, 96.0, 94.0, 92.0, 90.0])

        # Days 41-44: Sharp rise (fast MA crosses above slow MA at the end)
        prices.extend([95.0, 100.0, 105.0, 110.0])

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should detect bullish crossover
        assert signal == 1

    def test_bearish_crossover_signal(self):
        """Test detection of bearish MA crossover (fast crosses below slow)."""
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)

        # Create controlled data where crossover happens at the very end
        # Pattern: Flat, then rise, then sharp drop
        prices = [100.0] * 36  # Flat for 36 days

        # Days 36-40: Rise (fast MA will rise above slow MA)
        prices.extend([102.0, 104.0, 106.0, 108.0, 110.0])

        # Days 41-44: Sharp drop (fast MA crosses below slow MA at the end)
        prices.extend([105.0, 100.0, 95.0, 90.0])

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should detect bearish crossover
        assert signal == -1

    def test_no_signal_when_no_crossover(self):
        """Test that no signal is generated without crossover."""
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)

        # Flat market - no crossover
        data = create_sample_data(50, trend="flat", volatility=0.001)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        strategy = SimpleMAStrategy(fast_period=10, slow_period=30)

        # Only 20 rows (less than slow_period)
        data = create_sample_data(20)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_exact_minimum_data(self):
        """Test with exact minimum required data."""
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)

        # Exactly 10 rows
        data = create_sample_data(10)

        signal = strategy.analyze(data)

        # Should not crash, returns hold
        assert signal in [0, 1, -1]

    def test_varying_fast_slow_periods(self):
        """Test with different fast/slow period combinations."""
        data = create_sample_data(100, trend="up")

        # Short-term
        strategy1 = SimpleMAStrategy(fast_period=5, slow_period=15)
        signal1 = strategy1.analyze(data)
        assert signal1 in [0, 1, -1]

        # Long-term
        strategy2 = SimpleMAStrategy(fast_period=20, slow_period=50)
        signal2 = strategy2.analyze(data)
        assert signal2 in [0, 1, -1]


# ============================================================================
# Test: MomentumStrategy
# ============================================================================


class TestMomentumStrategy:
    """Test Momentum strategy."""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters."""
        strategy = MomentumStrategy()

        assert strategy.lookback == 20
        assert strategy.threshold == 0.02
        assert strategy.position == 0

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = MomentumStrategy(lookback=10, threshold=0.05)

        assert strategy.lookback == 10
        assert strategy.threshold == 0.05

    def test_get_required_periods(self):
        """Test required periods calculation."""
        strategy = MomentumStrategy(lookback=20)

        assert strategy.get_required_periods() == 21

    def test_positive_momentum_signal(self):
        """Test detection of positive momentum."""
        strategy = MomentumStrategy(lookback=10, threshold=0.02)

        # Create data with 5% gain over lookback period (exceeds 2% threshold)
        prices = [100.0] * 15  # Flat start
        prices.extend([100.0 + i * 0.5 for i in range(1, 7)])  # Then rise 3%

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should detect positive momentum
        assert signal == 1

    def test_negative_momentum_signal(self):
        """Test detection of negative momentum."""
        strategy = MomentumStrategy(lookback=10, threshold=0.02)

        # Create data with 5% loss over lookback period (exceeds 2% threshold)
        prices = [100.0] * 15  # Flat start
        prices.extend([100.0 - i * 0.5 for i in range(1, 7)])  # Then drop 3%

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should detect negative momentum
        assert signal == -1

    def test_weak_momentum_no_signal(self):
        """Test that weak momentum doesn't generate signal."""
        strategy = MomentumStrategy(lookback=20, threshold=0.05)

        # Flat market - weak momentum
        data = create_sample_data(50, trend="flat", volatility=0.001)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_momentum_at_threshold(self):
        """Test momentum exactly at threshold."""
        strategy = MomentumStrategy(lookback=10, threshold=0.02)

        data = create_sample_data(30, base_price=100.0)
        # Set up exactly 2% momentum
        data.loc[data.index[-11], "close"] = 100.0
        data.loc[data.index[-1], "close"] = 102.0

        signal = strategy.analyze(data)

        # Should not trigger (must be > threshold)
        assert signal == 0

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        strategy = MomentumStrategy(lookback=20)

        # Only 15 rows
        data = create_sample_data(15)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_different_thresholds(self):
        """Test with different momentum thresholds."""
        data = create_sample_data(50, trend="up")

        # Sensitive (low threshold)
        strategy1 = MomentumStrategy(lookback=10, threshold=0.01)
        signal1 = strategy1.analyze(data)

        # Conservative (high threshold)
        strategy2 = MomentumStrategy(lookback=10, threshold=0.10)
        signal2 = strategy2.analyze(data)

        # Low threshold more likely to signal
        assert signal1 in [0, 1, -1]
        assert signal2 in [0, 1, -1]


# ============================================================================
# Test: MeanReversionStrategy
# ============================================================================


class TestMeanReversionStrategy:
    """Test Mean Reversion strategy."""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters."""
        strategy = MeanReversionStrategy()

        assert strategy.period == 20
        assert strategy.num_std == 2.0
        assert strategy.position == 0

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = MeanReversionStrategy(period=30, num_std=2.5)

        assert strategy.period == 30
        assert strategy.num_std == 2.5

    def test_get_required_periods(self):
        """Test required periods calculation."""
        strategy = MeanReversionStrategy(period=20)

        assert strategy.get_required_periods() == 20

    def test_oversold_buy_signal(self):
        """Test buy signal when price is oversold (below lower band)."""
        strategy = MeanReversionStrategy(period=10, num_std=2.0)

        # Create data with stable prices then a sharp drop
        prices = [100.0] * 15  # Flat for stable mean/std
        prices.append(85.0)  # Sharp drop well below 2 std

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should signal buy (oversold)
        assert signal == 1

    def test_overbought_sell_signal(self):
        """Test sell signal when price is overbought (above upper band)."""
        strategy = MeanReversionStrategy(period=10, num_std=2.0)

        # Create oscillating data
        data = create_sample_data(30, base_price=100.0, trend="flat", volatility=0.02)
        # Force current price above upper band
        data.loc[data.index[-1], "close"] = 110.0

        signal = strategy.analyze(data)

        # Should signal sell (overbought)
        assert signal == -1

    def test_within_bands_no_signal(self):
        """Test no signal when price is within bands."""
        strategy = MeanReversionStrategy(period=20, num_std=2.0)

        # Create stable data
        data = create_sample_data(50, trend="flat", volatility=0.01)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        strategy = MeanReversionStrategy(period=20)

        # Only 15 rows
        data = create_sample_data(15)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_different_std_deviations(self):
        """Test with different standard deviation multipliers."""
        data = create_sample_data(50, base_price=100.0, volatility=0.02)
        data.loc[data.index[-1], "close"] = 95.0

        # Tight bands (1 std)
        strategy1 = MeanReversionStrategy(period=20, num_std=1.0)
        signal1 = strategy1.analyze(data)

        # Wide bands (3 std)
        strategy2 = MeanReversionStrategy(period=20, num_std=3.0)
        signal2 = strategy2.analyze(data)

        # Tight bands more likely to trigger
        assert signal1 in [0, 1, -1]
        assert signal2 in [0, 1, -1]

    def test_exact_band_boundary(self):
        """Test behavior at exact band boundaries."""
        strategy = MeanReversionStrategy(period=10, num_std=2.0)

        # Create data with tight range, then set last price well below
        prices = [100.0] * 15  # Flat for stable mean/std
        # Add one very low price at the end
        prices.append(90.0)  # Well below 2 std from mean of 100

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.01 for p in prices],
                "low": [p * 0.99 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should trigger buy (price well below lower band)
        assert signal == 1


# ============================================================================
# Test: VolatilityStrategy
# ============================================================================


class TestVolatilityStrategy:
    """Test Volatility strategy."""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters."""
        strategy = VolatilityStrategy()

        assert strategy.period == 20
        assert strategy.vol_threshold == 0.02
        assert strategy.position == 0

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = VolatilityStrategy(period=30, vol_threshold=0.03)

        assert strategy.period == 30
        assert strategy.vol_threshold == 0.03

    def test_get_required_periods(self):
        """Test required periods calculation."""
        strategy = VolatilityStrategy(period=20)

        assert strategy.get_required_periods() == 21

    def test_high_volatility_no_signal(self):
        """Test that high volatility prevents trading."""
        strategy = VolatilityStrategy(period=10, vol_threshold=0.01)

        # Create high volatility data
        data = create_sample_data(30, volatility=0.05)

        signal = strategy.analyze(data)

        # Should not trade in high volatility
        assert signal == 0

    def test_low_volatility_positive_momentum(self):
        """Test buy signal in low volatility with positive momentum."""
        strategy = VolatilityStrategy(period=10, vol_threshold=0.05)

        # Create very low volatility data with 2% gain over period
        prices = [100.0] * 15  # Flat start for low volatility
        prices.extend([100.0 + i * 0.25 for i in range(1, 7)])  # Gentle rise to 101.5

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.001 for p in prices],  # Very tight range
                "low": [p * 0.999 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should signal buy (low vol + positive momentum > 1%)
        assert signal == 1

    def test_low_volatility_negative_momentum(self):
        """Test sell signal in low volatility with negative momentum."""
        strategy = VolatilityStrategy(period=10, vol_threshold=0.05)

        # Create very low volatility data with 2% loss over period
        prices = [100.0] * 15  # Flat start for low volatility
        prices.extend([100.0 - i * 0.25 for i in range(1, 7)])  # Gentle drop to 98.5

        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * 1.001 for p in prices],  # Very tight range
                "low": [p * 0.999 for p in prices],
                "close": prices,
                "volume": [1000000] * len(prices),
            }
        )

        signal = strategy.analyze(data)

        # Should signal sell (low vol + negative momentum < -1%)
        assert signal == -1

    def test_low_volatility_weak_momentum(self):
        """Test no signal in low volatility with weak momentum."""
        strategy = VolatilityStrategy(period=20, vol_threshold=0.05)

        # Create low volatility flat market
        data = create_sample_data(50, trend="flat", volatility=0.001)

        signal = strategy.analyze(data)

        # Should not signal (momentum below 1%)
        assert signal == 0

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        strategy = VolatilityStrategy(period=20)

        # Only 15 rows
        data = create_sample_data(15)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_volatility_at_threshold(self):
        """Test behavior at volatility threshold."""
        strategy = VolatilityStrategy(period=10, vol_threshold=0.01)

        # Create data with higher volatility
        # Use volatility=0.02 which will exceed 0.01 threshold
        data = create_sample_data(30, volatility=0.02)

        signal = strategy.analyze(data)

        # Should not trade if vol >= threshold
        assert signal == 0

    def test_different_volatility_thresholds(self):
        """Test with different volatility thresholds."""
        data = create_sample_data(50, volatility=0.03)

        # Sensitive (high threshold - allows more volatility)
        strategy1 = VolatilityStrategy(period=20, vol_threshold=0.10)
        signal1 = strategy1.analyze(data)

        # Conservative (low threshold - requires low volatility)
        strategy2 = VolatilityStrategy(period=20, vol_threshold=0.01)
        signal2 = strategy2.analyze(data)

        # Both should be valid signals
        assert signal1 in [0, 1, -1]
        assert signal2 in [0, 1, -1]


# ============================================================================
# Test: BreakoutStrategy
# ============================================================================


class TestBreakoutStrategy:
    """Test Breakout strategy."""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters."""
        strategy = BreakoutStrategy()

        assert strategy.lookback == 20
        assert strategy.position == 0

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters."""
        strategy = BreakoutStrategy(lookback=30)

        assert strategy.lookback == 30

    def test_get_required_periods(self):
        """Test required periods calculation."""
        strategy = BreakoutStrategy(lookback=20)

        assert strategy.get_required_periods() == 21

    def test_upward_breakout_signal(self):
        """Test buy signal on upward breakout."""
        strategy = BreakoutStrategy(lookback=10)

        # Create data with breakout
        data = create_sample_data(30, base_price=100.0, trend="flat", volatility=0.01)
        # Force breakout - current price above recent high
        data.loc[data.index[-1], "close"] = 110.0

        signal = strategy.analyze(data)

        # Should signal buy (upward breakout)
        assert signal == 1

    def test_downward_breakout_signal(self):
        """Test sell signal on downward breakout."""
        strategy = BreakoutStrategy(lookback=10)

        # Create data with breakout
        data = create_sample_data(30, base_price=100.0, trend="flat", volatility=0.01)
        # Force breakout - current price below recent low
        data.loc[data.index[-1], "close"] = 90.0
        data.loc[data.index[-1], "low"] = 90.0

        signal = strategy.analyze(data)

        # Should signal sell (downward breakout)
        assert signal == -1

    def test_no_breakout_signal(self):
        """Test no signal when price is within range."""
        strategy = BreakoutStrategy(lookback=20)

        # Create stable range-bound data
        data = create_sample_data(50, trend="flat", volatility=0.01)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_insufficient_data(self):
        """Test behavior with insufficient data."""
        strategy = BreakoutStrategy(lookback=20)

        # Only 15 rows
        data = create_sample_data(15)

        signal = strategy.analyze(data)

        assert signal == 0

    def test_breakout_at_exact_high(self):
        """Test behavior when price equals recent high."""
        strategy = BreakoutStrategy(lookback=10)

        data = create_sample_data(30, base_price=100.0, volatility=0.01)
        recent_high = data["high"].iloc[-11:-1].max()
        data.loc[data.index[-1], "close"] = recent_high

        signal = strategy.analyze(data)

        # Should not trigger (must be > high)
        assert signal == 0

    def test_different_lookback_periods(self):
        """Test with different lookback periods."""
        data = create_sample_data(100, base_price=100.0, volatility=0.02)
        data.loc[data.index[-1], "close"] = 105.0

        # Short lookback
        strategy1 = BreakoutStrategy(lookback=5)
        signal1 = strategy1.analyze(data)

        # Long lookback
        strategy2 = BreakoutStrategy(lookback=50)
        signal2 = strategy2.analyze(data)

        # Both should produce valid signals
        assert signal1 in [0, 1, -1]
        assert signal2 in [0, 1, -1]

    def test_breakout_uses_high_low_not_close(self):
        """Test that breakout uses high/low columns correctly."""
        strategy = BreakoutStrategy(lookback=10)

        data = create_sample_data(30, base_price=100.0, volatility=0.01)

        # Ensure high column has the peaks
        data.loc[data.index[-15], "high"] = 105.0
        data.loc[data.index[-1], "close"] = 106.0

        signal = strategy.analyze(data)

        # Should detect breakout above high
        assert signal == 1


# ============================================================================
# Test: Strategy Factory
# ============================================================================


class TestStrategyFactory:
    """Test strategy factory function."""

    def test_create_simple_ma_strategy(self):
        """Test creating SimpleMAStrategy through factory."""
        strategy = create_paper_strategy("SimpleMAStrategy", fast_period=5, slow_period=15)

        assert isinstance(strategy, SimpleMAStrategy)
        assert strategy.fast_period == 5
        assert strategy.slow_period == 15

    def test_create_momentum_strategy(self):
        """Test creating MomentumStrategy through factory."""
        strategy = create_paper_strategy("MomentumStrategy", lookback=15, threshold=0.03)

        assert isinstance(strategy, MomentumStrategy)
        assert strategy.lookback == 15
        assert strategy.threshold == 0.03

    def test_create_mean_reversion_strategy(self):
        """Test creating MeanReversionStrategy through factory."""
        strategy = create_paper_strategy("MeanReversionStrategy", period=25, num_std=2.5)

        assert isinstance(strategy, MeanReversionStrategy)
        assert strategy.period == 25
        assert strategy.num_std == 2.5

    def test_create_volatility_strategy(self):
        """Test creating VolatilityStrategy through factory."""
        strategy = create_paper_strategy("VolatilityStrategy", period=15, vol_threshold=0.03)

        assert isinstance(strategy, VolatilityStrategy)
        assert strategy.period == 15
        assert strategy.vol_threshold == 0.03

    def test_create_breakout_strategy(self):
        """Test creating BreakoutStrategy through factory."""
        strategy = create_paper_strategy("BreakoutStrategy", lookback=25)

        assert isinstance(strategy, BreakoutStrategy)
        assert strategy.lookback == 25

    def test_create_strategy_with_default_params(self):
        """Test creating strategy with default parameters."""
        strategy = create_paper_strategy("SimpleMAStrategy")

        assert isinstance(strategy, SimpleMAStrategy)
        assert strategy.fast_period == 10
        assert strategy.slow_period == 30

    def test_create_invalid_strategy(self):
        """Test that invalid strategy name raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_paper_strategy("InvalidStrategy")

    def test_error_message_shows_available_strategies(self):
        """Test that error message includes available strategies."""
        with pytest.raises(ValueError, match="Available:"):
            create_paper_strategy("NonExistentStrategy")

    def test_create_all_strategies(self):
        """Test creating all available strategies."""
        strategies = [
            "SimpleMAStrategy",
            "MomentumStrategy",
            "MeanReversionStrategy",
            "VolatilityStrategy",
            "BreakoutStrategy",
        ]

        for strategy_name in strategies:
            strategy = create_paper_strategy(strategy_name)
            assert strategy is not None


# ============================================================================
# Test: Position Tracking
# ============================================================================


class TestPositionTracking:
    """Test position tracking in strategies."""

    def test_initial_position_is_zero(self):
        """Test that all strategies start with zero position."""
        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            assert strategy.position == 0

    def test_position_attribute_exists(self):
        """Test that position attribute exists on all strategies."""
        strategy = SimpleMAStrategy()

        assert hasattr(strategy, "position")


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestStrategyEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_strategies_with_minimal_data(self):
        """Test all strategies with minimal valid data."""
        data = create_sample_data(60)  # Enough for all strategies

        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            signal = strategy.analyze(data)
            assert signal in [0, 1, -1]

    def test_all_strategies_with_nan_data(self):
        """Test strategies handle NaN values gracefully."""
        data = create_sample_data(100)
        # Introduce NaN
        data.loc[data.index[50], "close"] = float("nan")

        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            signal = strategy.analyze(data)
            # Should handle NaN or return valid signal
            assert signal in [0, 1, -1] or pd.isna(signal)

    def test_strategies_with_constant_price(self):
        """Test strategies with constant price (no movement)."""
        data = create_sample_data(100)
        data["close"] = 100.0  # Constant price
        data["high"] = 100.0
        data["low"] = 100.0

        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            signal = strategy.analyze(data)
            # Most should return 0 (no signal)
            assert signal in [0, 1, -1]

    def test_strategies_with_extreme_volatility(self):
        """Test strategies with extreme volatility."""
        data = create_sample_data(100, volatility=0.5)

        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            signal = strategy.analyze(data)
            assert signal in [0, 1, -1]

    def test_strategies_with_zero_prices(self):
        """Test strategies with zero prices."""
        data = create_sample_data(100)
        data["close"] = 0.0

        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            signal = strategy.analyze(data)
            # Should handle gracefully
            assert signal in [0, 1, -1]

    def test_strategies_with_negative_prices(self):
        """Test strategies with negative prices (invalid but handled)."""
        data = create_sample_data(100)
        data["close"] = -100.0

        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            signal = strategy.analyze(data)
            # Should handle invalid data
            assert signal in [0, 1, -1]

    def test_strategies_with_single_row(self):
        """Test strategies with only one data row."""
        data = create_sample_data(1)

        strategies = [
            SimpleMAStrategy(),
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolatilityStrategy(),
            BreakoutStrategy(),
        ]

        for strategy in strategies:
            signal = strategy.analyze(data)
            # Should return hold (insufficient data)
            assert signal == 0

    def test_strategies_params_stored_correctly(self):
        """Test that strategy parameters are stored in params dict."""
        strategy = SimpleMAStrategy(fast_period=7, slow_period=21)

        assert strategy.params["fast_period"] == 7
        assert strategy.params["slow_period"] == 21

    def test_strategies_with_extra_kwargs(self):
        """Test strategies accept extra kwargs without error."""
        # Should not raise error even with extra params
        strategy = SimpleMAStrategy(fast_period=10, slow_period=30, extra_param="ignored")

        assert "extra_param" in strategy.params
        assert strategy.fast_period == 10


# ============================================================================
# Test: Integration Scenarios
# ============================================================================


class TestStrategyIntegrationScenarios:
    """Test realistic trading scenarios."""

    def test_trend_following_with_ma(self):
        """Test MA strategy in trending market."""
        strategy = SimpleMAStrategy(fast_period=5, slow_period=10)

        # Strong uptrend
        data = create_sample_data(50, trend="up", volatility=0.01)
        signal = strategy.analyze(data)

        # Should eventually signal buy or hold
        assert signal in [0, 1]

    def test_momentum_in_volatile_market(self):
        """Test momentum strategy in volatile market."""
        strategy = MomentumStrategy(lookback=10, threshold=0.05)

        # High volatility
        data = create_sample_data(50, volatility=0.05)
        signal = strategy.analyze(data)

        # Should handle volatility
        assert signal in [0, 1, -1]

    def test_mean_reversion_in_oscillating_market(self):
        """Test mean reversion in oscillating market."""
        strategy = MeanReversionStrategy(period=20, num_std=2.0)

        # Oscillating prices
        data = create_sample_data(100, trend="oscillating", volatility=0.02)
        signal = strategy.analyze(data)

        # Should generate signals at extremes
        assert signal in [0, 1, -1]

    def test_breakout_after_consolidation(self):
        """Test breakout detection after consolidation."""
        strategy = BreakoutStrategy(lookback=20)

        # Consolidation followed by breakout
        data = create_sample_data(50, trend="flat", volatility=0.005)
        # Force breakout
        data.loc[data.index[-1], "close"] = data["close"].max() * 1.05

        signal = strategy.analyze(data)

        # Should detect breakout
        assert signal == 1

    def test_volatility_filter_effectiveness(self):
        """Test that volatility filter prevents trading in volatile conditions."""
        strategy = VolatilityStrategy(period=10, vol_threshold=0.01)

        # Create high volatility with strong trend
        data = create_sample_data(50, trend="up", volatility=0.10)
        signal = strategy.analyze(data)

        # Should not trade despite trend
        assert signal == 0
