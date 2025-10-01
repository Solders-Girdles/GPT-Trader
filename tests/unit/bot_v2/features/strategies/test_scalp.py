"""Tests for scalp strategy"""

import pytest
from bot_v2.features.strategies.scalp import ScalpStrategy
from bot_v2.features.strategies.interfaces import StrategyContext


class TestScalpStrategy:
    """Test suite for ScalpStrategy"""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters"""
        strategy = ScalpStrategy()

        assert strategy.name == "scalp"
        assert strategy.bp_threshold == 0.0005

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters"""
        strategy = ScalpStrategy(bp_threshold=0.001)

        assert strategy.bp_threshold == 0.001

    def test_no_signals_insufficient_data(self):
        """Test no signals when insufficient price history"""
        strategy = ScalpStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Only 1 price, need at least 2
        strategy.price_history["BTC-USD"] = [100.0]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_buy_signal_at_threshold(self):
        """Test BUY signal above threshold"""
        strategy = ScalpStrategy(bp_threshold=0.001)  # 0.1%
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price increases by 0.11% (slightly above threshold)
        strategy.price_history["BTC-USD"] = [100.0, 100.11]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        assert signals[0].confidence == 1.0

    def test_buy_signal_above_threshold(self):
        """Test BUY signal above threshold with scaled confidence"""
        strategy = ScalpStrategy(bp_threshold=0.001)  # 0.1%
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price increases by 0.2% (2x threshold)
        strategy.price_history["BTC-USD"] = [100.0, 100.2]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        # Confidence capped at 1.0 even though change is 2x threshold
        assert signals[0].confidence == 1.0

    def test_sell_signal_at_threshold(self):
        """Test SELL signal above threshold"""
        strategy = ScalpStrategy(bp_threshold=0.001)  # 0.1%
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price decreases by 0.11% (slightly above threshold)
        strategy.price_history["BTC-USD"] = [100.0, 99.89]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"
        assert signals[0].confidence == 1.0

    def test_sell_signal_below_threshold(self):
        """Test SELL signal below threshold with scaled confidence"""
        strategy = ScalpStrategy(bp_threshold=0.001)  # 0.1%
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price decreases by 0.15% (1.5x threshold)
        strategy.price_history["BTC-USD"] = [100.0, 99.85]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"
        # Confidence capped at 1.0
        assert signals[0].confidence == 1.0

    def test_no_signal_below_threshold(self):
        """Test no signal when change is below threshold"""
        strategy = ScalpStrategy(bp_threshold=0.001)  # 0.1%
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price increases by only 0.05% (half threshold)
        strategy.price_history["BTC-USD"] = [100.0, 100.05]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_confidence_scaling(self):
        """Test confidence scales with price change magnitude"""
        strategy = ScalpStrategy(bp_threshold=0.002)  # 0.2%
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price increases by 0.3% (1.5x threshold, but capped at 1.0)
        strategy.price_history["BTC-USD"] = [100.0, 100.3]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        # Confidence is min(1.0, 0.003/0.002) = min(1.0, 1.5) = 1.0
        assert signals[0].confidence == 1.0

    def test_multiple_symbols(self):
        """Test signals for multiple symbols"""
        strategy = ScalpStrategy(bp_threshold=0.001)
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD", "SOL-USD"])

        # BTC: buy signal (0.2% increase)
        strategy.price_history["BTC-USD"] = [100.0, 100.2]

        # ETH: sell signal (0.2% decrease)
        strategy.price_history["ETH-USD"] = [200.0, 199.6]

        # SOL: no signal (below threshold)
        strategy.price_history["SOL-USD"] = [50.0, 50.03]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 2
        btc_signal = next(s for s in signals if s.symbol == "BTC-USD")
        eth_signal = next(s for s in signals if s.symbol == "ETH-USD")
        assert btc_signal.side == "buy"
        assert eth_signal.side == "sell"

    def test_empty_price_history(self):
        """Test with no price history"""
        strategy = ScalpStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_zero_last_price(self):
        """Test handling of zero last price"""
        strategy = ScalpStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Last price is zero (invalid)
        strategy.price_history["BTC-USD"] = [0.0, 100.0]

        signals = strategy.get_signals(ctx)

        # Should skip this symbol
        assert len(signals) == 0

    def test_negative_last_price(self):
        """Test handling of negative last price"""
        strategy = ScalpStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Last price is negative (invalid)
        strategy.price_history["BTC-USD"] = [-100.0, 100.0]

        signals = strategy.get_signals(ctx)

        # Should skip this symbol
        assert len(signals) == 0

    def test_small_bp_threshold(self):
        """Test with very small threshold (tick scalping)"""
        strategy = ScalpStrategy(bp_threshold=0.0001)  # 1 bp = 0.01%
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price increases by 0.02%
        strategy.price_history["BTC-USD"] = [100.0, 100.02]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].side == "buy"

    def test_large_price_movement(self):
        """Test with large price movement"""
        strategy = ScalpStrategy(bp_threshold=0.001)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Price increases by 5%
        strategy.price_history["BTC-USD"] = [100.0, 105.0]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].side == "buy"
        # Confidence capped at 1.0 even for large moves
        assert signals[0].confidence == 1.0

    def test_exact_zero_change(self):
        """Test no signal when price doesn't change"""
        strategy = ScalpStrategy(bp_threshold=0.001)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # No price change
        strategy.price_history["BTC-USD"] = [100.0, 100.0]

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_multiple_price_points(self):
        """Test uses only last two prices"""
        strategy = ScalpStrategy(bp_threshold=0.001)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Multiple prices, only last two matter
        strategy.price_history["BTC-USD"] = [95.0, 96.0, 97.0, 100.0, 100.15]

        signals = strategy.get_signals(ctx)

        # Should compare 100.0 vs 100.15
        assert len(signals) == 1
        assert signals[0].side == "buy"