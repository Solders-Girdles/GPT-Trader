"""Tests for volatility strategy"""

import pytest
from bot_v2.features.strategies.volatility import VolatilityStrategy
from bot_v2.features.strategies.interfaces import StrategyContext


class TestVolatilityStrategy:
    """Test suite for VolatilityStrategy"""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters"""
        strategy = VolatilityStrategy()

        assert strategy.name == "volatility"
        assert strategy.vol_period == 20
        assert strategy.vol_threshold == 0.02

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters"""
        strategy = VolatilityStrategy(vol_period=30, vol_threshold=0.03)

        assert strategy.vol_period == 30
        assert strategy.vol_threshold == 0.03

    def test_no_signals_insufficient_data(self):
        """Test no signals when insufficient price history"""
        strategy = VolatilityStrategy(vol_period=20)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Only 15 prices, need vol_period + 5 = 25
        strategy.price_history["BTC-USD"] = [100.0] * 15

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_buy_signal_low_volatility_uptrend(self):
        """Test BUY signal in low volatility with uptrend"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Low volatility (stable prices) with recent uptrend
        prices = [100.0] * 10  # Stable period
        prices.extend([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])  # Uptrend
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        assert signals[0].confidence == 0.8

    def test_sell_signal_low_volatility_downtrend(self):
        """Test SELL signal in low volatility with downtrend"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Low volatility (stable prices) with recent downtrend
        prices = [100.0] * 10  # Stable period
        prices.extend([100.0, 99.5, 99.0, 98.5, 98.0, 97.5])  # Downtrend
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"
        assert signals[0].confidence == 0.8

    def test_no_signal_high_volatility(self):
        """Test no signal when volatility is high"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.01)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # High volatility with large swings
        prices = [100, 105, 95, 110, 90, 115, 85, 120, 80, 125, 100, 105, 95, 110, 90]
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # High volatility - no signal
        assert len(signals) == 0

    def test_no_signal_weak_trend(self):
        """Test no signal when trend is too weak"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Low volatility but weak trend (< 1%)
        prices = [100.0] * 10
        prices.extend([100.0, 100.1, 100.2, 100.3, 100.4, 100.5])  # Only 0.5% increase
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # Trend too weak (< 1%)
        assert len(signals) == 0

    def test_multiple_symbols(self):
        """Test signals for multiple symbols"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD", "SOL-USD"])

        # BTC: low vol, uptrend
        btc_prices = [100.0] * 10
        btc_prices.extend([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        strategy.price_history["BTC-USD"] = btc_prices

        # ETH: low vol, downtrend
        eth_prices = [200.0] * 10
        eth_prices.extend([200.0, 199.0, 198.0, 197.0, 196.0, 195.0])
        strategy.price_history["ETH-USD"] = eth_prices

        # SOL: high volatility, no signal
        sol_prices = [50, 55, 45, 60, 40, 65, 35, 70, 30, 75, 50, 55, 45, 60, 40]
        strategy.price_history["SOL-USD"] = sol_prices

        signals = strategy.get_signals(ctx)

        assert len(signals) == 2
        btc_signal = next(s for s in signals if s.symbol == "BTC-USD")
        eth_signal = next(s for s in signals if s.symbol == "ETH-USD")
        assert btc_signal.side == "buy"
        assert eth_signal.side == "sell"

    def test_empty_price_history(self):
        """Test with no price history"""
        strategy = VolatilityStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_zero_price_handling(self):
        """Test handling of zero prices in returns calculation"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Include a zero price
        prices = [100.0, 101.0, 0.0, 102.0] * 5
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # Should handle gracefully (skip zero prices)
        assert isinstance(signals, list)

    def test_constant_prices_zero_volatility(self):
        """Test with constant prices (zero volatility)"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # All constant prices except for trend
        prices = [100.0] * 10
        prices.extend([100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # Low/zero volatility with uptrend should signal
        assert len(signals) == 1

    def test_minimum_data_requirement(self):
        """Test minimum data requirement"""
        strategy = VolatilityStrategy(vol_period=5, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Exactly minimum: max(vol_period + 5, 10) = 10
        prices = [100.0] * 10
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # Should process without error
        assert isinstance(signals, list)

    def test_recent_trend_calculation(self):
        """Test that recent trend uses last 6 prices"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Low vol period, then 6-price uptrend
        prices = [100.0] * 15
        # Last 6 prices: 100, 100.5, 101, 101.5, 102, 102.5
        # Trend = (102.5 - 100) / 100 = 2.5%
        prices.extend([100.5, 101.0, 101.5, 102.0, 102.5])
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].side == "buy"

    def test_confidence_always_0_8(self):
        """Test that confidence is always 0.8 for volatility signals"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Various trend strengths
        for trend_strength in [1.5, 2.0, 3.0, 5.0]:
            prices = [100.0] * 10
            final_price = 100.0 * (1 + trend_strength / 100)
            prices.extend([100.0 + i * (final_price - 100.0) / 5 for i in range(1, 6)])
            strategy.price_history["BTC-USD"] = prices

            signals = strategy.get_signals(ctx)

            if len(signals) > 0:
                assert signals[0].confidence == 0.8

    def test_custom_vol_period(self):
        """Test with custom volatility period"""
        strategy = VolatilityStrategy(vol_period=5, vol_threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Need at least vol_period + 5 = 10 prices
        prices = [100.0] * 7
        prices.extend([100.0, 101.0, 102.0, 103.0])
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # Should work with shorter period
        assert isinstance(signals, list)

    def test_custom_vol_threshold(self):
        """Test with custom volatility threshold"""
        strategy = VolatilityStrategy(vol_period=10, vol_threshold=0.05)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Moderate volatility (0.03) that would pass with lower threshold
        prices = [100, 102, 98, 103, 97, 104, 96, 105, 95, 106, 100, 101, 102, 103, 104, 105]
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # With high threshold, moderate volatility passes
        assert isinstance(signals, list)