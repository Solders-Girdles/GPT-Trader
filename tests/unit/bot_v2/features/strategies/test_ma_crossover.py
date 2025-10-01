"""Tests for MA crossover strategy"""

import pytest
from bot_v2.features.strategies.ma_crossover import MAStrategy
from bot_v2.features.strategies.interfaces import StrategyContext


class TestMAStrategy:
    """Test suite for MAStrategy"""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters"""
        strategy = MAStrategy()

        assert strategy.name == "ma_crossover"
        assert strategy.fast_period == 10
        assert strategy.slow_period == 30
        assert strategy.lookback == 32

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters"""
        strategy = MAStrategy(fast_period=5, slow_period=20)

        assert strategy.fast_period == 5
        assert strategy.slow_period == 20
        assert strategy.lookback == 22

    def test_no_signals_insufficient_data(self):
        """Test no signals when insufficient price history"""
        strategy = MAStrategy(fast_period=5, slow_period=10)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Only 8 prices, need slow_period + 1 = 11
        strategy.price_history["BTC-USD"] = [100.0] * 8

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_buy_signal_bullish_crossover(self):
        """Test BUY signal on bullish MA crossover"""
        strategy = MAStrategy(fast_period=2, slow_period=4)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Build prices where fast MA crosses above slow MA
        # Need: prev_fast <= prev_slow and fast > slow
        # prices: [95, 100, 105, 105, 105, 110, 120]
        # prev_fast (105,110) = 107.5, prev_slow (105,105,105,110) = 106.25
        # fast (110,120) = 115, slow (105,105,110,120) = 110
        # Actually let's ensure clean crossover
        prices = [100, 100, 100, 100, 95, 105, 125]
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # Should generate buy signal on crossover
        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        assert signals[0].confidence == 1.0

    def test_sell_signal_bearish_crossover(self):
        """Test SELL signal on bearish MA crossover"""
        strategy = MAStrategy(fast_period=2, slow_period=4)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Build prices where fast MA crosses below slow MA
        # Need: prev_fast >= prev_slow and fast < slow
        prices = [120, 120, 120, 120, 125, 115, 95]
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"

    def test_no_signal_when_already_crossed(self):
        """Test no signal when MAs already crossed in previous period"""
        strategy = MAStrategy(fast_period=3, slow_period=5)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Fast already above slow, no new crossover
        prices = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        # No crossover happening, so no signal
        assert len(signals) == 0

    def test_no_signal_parallel_movement(self):
        """Test no signal when MAs move in parallel"""
        strategy = MAStrategy(fast_period=3, slow_period=5)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Steady increase, MAs move parallel
        prices = [100.0 + i for i in range(20)]
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_multiple_symbols(self):
        """Test signals for multiple symbols"""
        strategy = MAStrategy(fast_period=2, slow_period=4)
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD"])

        # BTC: bullish crossover
        btc_prices = [100, 100, 100, 100, 95, 105, 125]
        strategy.price_history["BTC-USD"] = btc_prices

        # ETH: bearish crossover
        eth_prices = [200, 200, 200, 200, 205, 195, 175]
        strategy.price_history["ETH-USD"] = eth_prices

        signals = strategy.get_signals(ctx)

        assert len(signals) == 2
        btc_signal = next(s for s in signals if s.symbol == "BTC-USD")
        eth_signal = next(s for s in signals if s.symbol == "ETH-USD")
        assert btc_signal.side == "buy"
        assert eth_signal.side == "sell"

    def test_exact_minimum_data(self):
        """Test with exactly slow_period + 1 prices"""
        strategy = MAStrategy(fast_period=3, slow_period=5)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Exactly 6 prices (slow_period + 1)
        prices = [100, 100, 100, 110, 120, 130]
        strategy.price_history["BTC-USD"] = prices

        # Should be able to compute signals
        signals = strategy.get_signals(ctx)

        # May or may not signal depending on crossover
        assert isinstance(signals, list)

    def test_empty_price_history(self):
        """Test with no price history"""
        strategy = MAStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_symbols_with_partial_data(self):
        """Test with some symbols having data and others not"""
        strategy = MAStrategy(fast_period=2, slow_period=4)
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD"])

        # BTC has enough data with crossover
        btc_prices = [100, 100, 100, 100, 95, 105, 125]
        strategy.price_history["BTC-USD"] = btc_prices

        # ETH has insufficient data
        strategy.price_history["ETH-USD"] = [200.0] * 4

        signals = strategy.get_signals(ctx)

        # Only BTC should have a signal
        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"

    def test_lookback_calculation(self):
        """Test lookback is correctly calculated"""
        strategy1 = MAStrategy(fast_period=5, slow_period=20)
        strategy2 = MAStrategy(fast_period=15, slow_period=10)

        # Lookback should be max(fast, slow) + 2
        assert strategy1.lookback == 22
        assert strategy2.lookback == 17