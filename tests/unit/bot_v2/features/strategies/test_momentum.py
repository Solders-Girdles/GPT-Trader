"""Tests for momentum strategy"""

import pytest
from bot_v2.features.strategies.momentum import MomentumStrategy
from bot_v2.features.strategies.interfaces import StrategyContext


class TestMomentumStrategy:
    """Test suite for MomentumStrategy"""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters"""
        strategy = MomentumStrategy()

        assert strategy.name == "momentum"
        assert strategy.momentum_period == 10
        assert strategy.threshold == 0.02

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters"""
        strategy = MomentumStrategy(momentum_period=20, threshold=0.05)

        assert strategy.momentum_period == 20
        assert strategy.threshold == 0.05

    def test_update_price(self):
        """Test price update"""
        strategy = MomentumStrategy()

        strategy.update_price("BTC-USD", 50000.0)
        strategy.update_price("BTC-USD", 51000.0)

        assert len(strategy.price_history["BTC-USD"]) == 2
        assert strategy.price_history["BTC-USD"] == [50000.0, 51000.0]

    def test_get_signals_insufficient_data(self):
        """Test signal generation with insufficient price data"""
        strategy = MomentumStrategy(momentum_period=10)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add only 10 prices (need 11)
        for i in range(10):
            strategy.update_price("BTC-USD", 50000.0 + i * 100)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_get_signals_buy_strong_momentum(self):
        """Test buy signal generation with strong positive momentum"""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices with strong upward momentum
        base_price = 50000.0
        for i in range(15):
            strategy.update_price("BTC-USD", base_price + i * 200)  # ~4% increase

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        assert signals[0].confidence > 1.0  # Strong momentum

    def test_get_signals_buy_weak_momentum(self):
        """Test buy signal with weak positive momentum"""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices with weak upward momentum (just above threshold)
        base_price = 50000.0
        for i in range(15):
            strategy.update_price("BTC-USD", base_price + i * 105)  # ~2.1% increase over 11 periods

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].side == "buy"
        assert 1.0 <= signals[0].confidence <= 1.5

    def test_get_signals_sell_strong_momentum(self):
        """Test sell signal generation with strong negative momentum"""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices with strong downward momentum
        base_price = 50000.0
        for i in range(15):
            strategy.update_price("BTC-USD", base_price - i * 200)  # ~4% decrease

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"
        assert signals[0].confidence > 1.0

    def test_get_signals_no_momentum(self):
        """Test no signal with momentum below threshold"""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices with minimal momentum (< 2%)
        base_price = 50000.0
        for i in range(15):
            strategy.update_price("BTC-USD", base_price + i * 50)  # ~1% increase

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_get_signals_zero_old_price(self):
        """Test signal generation handles zero old price"""
        strategy = MomentumStrategy(momentum_period=10)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add zero price in history
        strategy.update_price("BTC-USD", 0.0)
        for i in range(15):
            strategy.update_price("BTC-USD", 50000.0 + i * 100)

        signals = strategy.get_signals(ctx)

        # Should handle gracefully without division by zero
        assert isinstance(signals, list)

    def test_get_signals_multiple_symbols(self):
        """Test signal generation for multiple symbols"""
        strategy = MomentumStrategy(momentum_period=5, threshold=0.03)
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD"])

        # BTC with positive momentum (3.3% over 6 periods)
        for i in range(10):
            strategy.update_price("BTC-USD", 50000.0 + i * 350)

        # ETH with negative momentum (-3.5% over 6 periods)
        for i in range(10):
            strategy.update_price("ETH-USD", 3000.0 - i * 18)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 2
        btc_signal = [s for s in signals if s.symbol == "BTC-USD"][0]
        eth_signal = [s for s in signals if s.symbol == "ETH-USD"][0]

        assert btc_signal.side == "buy"
        assert eth_signal.side == "sell"

    def test_get_signals_flat_prices(self):
        """Test signal generation with flat prices"""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add constant prices
        for _ in range(15):
            strategy.update_price("BTC-USD", 50000.0)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_confidence_calculation(self):
        """Test confidence calculation scales correctly"""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices with momentum at exactly 2x threshold
        base_price = 50000.0
        for i in range(15):
            strategy.update_price("BTC-USD", base_price + i * 150)  # ~4% (2x threshold)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        # Confidence should be capped at 1.0 or scale appropriately
        assert signals[0].confidence >= 1.0

    def test_price_history_capping(self):
        """Test price history is capped to prevent unbounded growth"""
        strategy = MomentumStrategy(lookback=100)

        # Add more prices than lookback
        for i in range(150):
            strategy.update_price("BTC-USD", 50000.0 + i)

        # Should be capped at lookback value
        assert len(strategy.price_history["BTC-USD"]) == 100

    def test_momentum_period_parameter_validation(self):
        """Test momentum period is converted to int"""
        strategy = MomentumStrategy(momentum_period="15")

        assert isinstance(strategy.momentum_period, int)
        assert strategy.momentum_period == 15

    def test_threshold_parameter_validation(self):
        """Test threshold is converted to float"""
        strategy = MomentumStrategy(threshold="0.05")

        assert isinstance(strategy.threshold, float)
        assert strategy.threshold == 0.05