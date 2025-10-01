"""Tests for breakout strategy"""

import pytest
from bot_v2.features.strategies.breakout import BreakoutStrategy
from bot_v2.features.strategies.interfaces import StrategyContext


@pytest.fixture
def breakout_strategy():
    """Create breakout strategy instance"""
    return BreakoutStrategy(breakout_period=20, threshold_pct=0.01)


@pytest.fixture
def strategy_context():
    """Create strategy context"""
    return StrategyContext(symbols=["BTC-USD", "ETH-USD"])


class TestBreakoutStrategy:
    """Test suite for BreakoutStrategy"""

    def test_initialization(self, breakout_strategy):
        """Test strategy initialization"""
        assert breakout_strategy.name == "breakout"
        assert breakout_strategy.breakout_period == 20
        assert breakout_strategy.threshold_pct == 0.01

    def test_initialization_with_defaults(self):
        """Test initialization with default parameters"""
        strategy = BreakoutStrategy()
        assert strategy.breakout_period == 20
        assert strategy.threshold_pct == 0.01

    def test_no_signals_insufficient_data(self, breakout_strategy, strategy_context):
        """Test no signals when insufficient price history"""
        # Only 10 prices, need 20
        breakout_strategy.price_history["BTC-USD"] = [100.0] * 10

        signals = breakout_strategy.get_signals(strategy_context)

        assert len(signals) == 0

    def test_buy_signal_upward_breakout(self, breakout_strategy, strategy_context):
        """Test BUY signal on upward breakout"""
        # Build price history with recent high at 1000
        prices = [1000.0] * 19
        # Current price breaks above recent high by 1.5%
        prices.append(1015.0)
        breakout_strategy.price_history["BTC-USD"] = prices

        signals = breakout_strategy.get_signals(strategy_context)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        assert signals[0].confidence == 1.0

    def test_sell_signal_downward_breakout(self, breakout_strategy, strategy_context):
        """Test SELL signal on downward breakout"""
        # Build price history with recent low at 1000
        prices = [1000.0] * 19
        # Current price breaks below recent low by 1.5%
        prices.append(985.0)
        breakout_strategy.price_history["BTC-USD"] = prices

        signals = breakout_strategy.get_signals(strategy_context)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"
        assert signals[0].confidence == 1.0

    def test_no_signal_within_threshold(self, breakout_strategy, strategy_context):
        """Test no signal when price is within threshold"""
        # Recent high at 1000, current price only 0.5% above
        prices = [1000.0] * 19
        prices.append(1005.0)
        breakout_strategy.price_history["BTC-USD"] = prices

        signals = breakout_strategy.get_signals(strategy_context)

        assert len(signals) == 0

    def test_no_signal_at_exactly_threshold(self, breakout_strategy, strategy_context):
        """Test no signal at exactly threshold boundary"""
        prices = [1000.0] * 19
        # Exactly at threshold (1% above)
        prices.append(1010.0)
        breakout_strategy.price_history["BTC-USD"] = prices

        signals = breakout_strategy.get_signals(strategy_context)

        # Should not signal - needs to be > threshold
        assert len(signals) == 0

    def test_multiple_symbols(self, breakout_strategy, strategy_context):
        """Test signals for multiple symbols"""
        # BTC breaks upward
        btc_prices = [1000.0] * 19
        btc_prices.append(1015.0)
        breakout_strategy.price_history["BTC-USD"] = btc_prices

        # ETH breaks downward
        eth_prices = [2000.0] * 19
        eth_prices.append(1970.0)
        breakout_strategy.price_history["ETH-USD"] = eth_prices

        signals = breakout_strategy.get_signals(strategy_context)

        assert len(signals) == 2
        btc_signal = next(s for s in signals if s.symbol == "BTC-USD")
        eth_signal = next(s for s in signals if s.symbol == "ETH-USD")
        assert btc_signal.side == "buy"
        assert eth_signal.side == "sell"

    def test_varying_prices_in_window(self, breakout_strategy, strategy_context):
        """Test breakout with varying prices in window"""
        # Prices vary between 900-1100
        prices = [900.0, 950.0, 1000.0, 1100.0, 1050.0] * 4
        # Current price breaks above 1100
        prices.append(1111.1)
        breakout_strategy.price_history["BTC-USD"] = prices

        signals = breakout_strategy.get_signals(strategy_context)

        assert len(signals) == 1
        assert signals[0].side == "buy"

    def test_custom_breakout_period(self, strategy_context):
        """Test with custom breakout period"""
        strategy = BreakoutStrategy(breakout_period=10, threshold_pct=0.01)

        # Only 11 prices, enough for period=10
        prices = [1000.0] * 10
        prices.append(1015.0)
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(strategy_context)

        assert len(signals) == 1
        assert signals[0].side == "buy"

    def test_custom_threshold(self, strategy_context):
        """Test with custom threshold percentage"""
        strategy = BreakoutStrategy(breakout_period=20, threshold_pct=0.05)

        # Price is 2% above recent high (below 5% threshold)
        prices = [1000.0] * 19
        prices.append(1020.0)
        strategy.price_history["BTC-USD"] = prices

        signals = strategy.get_signals(strategy_context)

        # Should not signal with 5% threshold
        assert len(signals) == 0

    def test_empty_price_history(self, breakout_strategy, strategy_context):
        """Test with no price history"""
        signals = breakout_strategy.get_signals(strategy_context)
        assert len(signals) == 0

    def test_exact_breakout_period_length(self, breakout_strategy, strategy_context):
        """Test with exactly breakout_period prices"""
        # Exactly 20 prices
        prices = [1000.0] * 19
        prices.append(1015.0)
        breakout_strategy.price_history["BTC-USD"] = prices

        signals = breakout_strategy.get_signals(strategy_context)

        assert len(signals) == 1

    def test_symbols_with_partial_data(self, breakout_strategy, strategy_context):
        """Test with some symbols having data and others not"""
        # BTC has enough data
        btc_prices = [1000.0] * 19
        btc_prices.append(1015.0)
        breakout_strategy.price_history["BTC-USD"] = btc_prices

        # ETH has insufficient data
        breakout_strategy.price_history["ETH-USD"] = [2000.0] * 5

        signals = breakout_strategy.get_signals(strategy_context)

        # Only BTC should have a signal
        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
