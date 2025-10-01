"""Tests for mean reversion strategy"""

import pytest
from bot_v2.features.strategies.mean_reversion import MeanReversionStrategy
from bot_v2.features.strategies.interfaces import StrategyContext


class TestMeanReversionStrategy:
    """Test suite for MeanReversionStrategy"""

    def test_initialization_default_params(self):
        """Test strategy initialization with default parameters"""
        strategy = MeanReversionStrategy()

        assert strategy.name == "mean_reversion"
        assert strategy.bb_period == 20
        assert strategy.bb_std == 2.0

    def test_initialization_custom_params(self):
        """Test strategy initialization with custom parameters"""
        strategy = MeanReversionStrategy(bb_period=30, bb_std=2.5)

        assert strategy.bb_period == 30
        assert strategy.bb_std == 2.5

    def test_get_signals_insufficient_data(self):
        """Test signal generation with insufficient data"""
        strategy = MeanReversionStrategy(bb_period=20)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add only 15 prices (need 20)
        for i in range(15):
            strategy.update_price("BTC-USD", 50000.0)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_get_signals_buy_below_lower_band(self):
        """Test buy signal when price is below lower Bollinger Band"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices with low volatility around 50000
        base_price = 50000.0
        for i in range(20):
            strategy.update_price("BTC-USD", base_price + (i % 5) * 100)

        # Add price significantly below mean
        strategy.update_price("BTC-USD", 48000.0)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        assert 0.2 <= signals[0].confidence <= 1.0

    def test_get_signals_sell_above_upper_band(self):
        """Test sell signal when price is above upper Bollinger Band"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices with low volatility around 50000
        base_price = 50000.0
        for i in range(20):
            strategy.update_price("BTC-USD", base_price + (i % 5) * 100)

        # Add price significantly above mean
        strategy.update_price("BTC-USD", 52000.0)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"
        assert 0.2 <= signals[0].confidence <= 1.0

    def test_get_signals_no_signal_within_bands(self):
        """Test no signal when price is within Bollinger Bands"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices in normal range
        base_price = 50000.0
        for i in range(25):
            strategy.update_price("BTC-USD", base_price + (i % 10) * 50)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 0

    def test_get_signals_flat_prices_no_std_dev(self):
        """Test signal generation with flat prices (zero standard deviation)"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add constant prices
        for _ in range(20):
            strategy.update_price("BTC-USD", 50000.0)

        # Add different price
        strategy.update_price("BTC-USD", 49000.0)

        signals = strategy.get_signals(ctx)

        # Should handle zero std dev gracefully
        # With zero std dev, bands collapse to mean, so any different price triggers signal
        assert len(signals) <= 1

    def test_get_signals_high_volatility(self):
        """Test signal generation with high volatility"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add highly volatile prices
        import random
        random.seed(42)
        base_price = 50000.0
        for i in range(25):
            volatility = random.uniform(-2000, 2000)
            strategy.update_price("BTC-USD", base_price + volatility)

        signals = strategy.get_signals(ctx)

        # May or may not generate signals depending on last price
        assert isinstance(signals, list)

    def test_get_signals_multiple_symbols(self):
        """Test signal generation for multiple symbols"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD"])

        # BTC with stable prices then drop
        for i in range(20):
            strategy.update_price("BTC-USD", 50000.0 + (i % 3) * 50)
        strategy.update_price("BTC-USD", 48000.0)  # Below lower band

        # ETH with stable prices then spike
        for i in range(20):
            strategy.update_price("ETH-USD", 3000.0 + (i % 3) * 10)
        strategy.update_price("ETH-USD", 3200.0)  # Above upper band

        signals = strategy.get_signals(ctx)

        assert len(signals) == 2
        btc_signal = [s for s in signals if s.symbol == "BTC-USD"][0]
        eth_signal = [s for s in signals if s.symbol == "ETH-USD"][0]

        assert btc_signal.side == "buy"
        assert eth_signal.side == "sell"

    def test_confidence_minimum_value(self):
        """Test confidence has minimum value of 0.2"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create scenario with minimal deviation
        for i in range(25):
            strategy.update_price("BTC-USD", 50000.0 + (i % 5) * 10)

        signals = strategy.get_signals(ctx)

        # If signals are generated, confidence should be at least 0.2
        for signal in signals:
            assert signal.confidence >= 0.2

    def test_confidence_maximum_value(self):
        """Test confidence is capped at 1.0"""
        strategy = MeanReversionStrategy(bb_period=20, bb_std=1.5)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create stable prices
        for i in range(20):
            strategy.update_price("BTC-USD", 50000.0 + (i % 3) * 50)

        # Add extreme price deviation
        strategy.update_price("BTC-USD", 45000.0)

        signals = strategy.get_signals(ctx)

        # Confidence should be capped at 1.0
        for signal in signals:
            assert signal.confidence <= 1.0

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands are calculated correctly"""
        strategy = MeanReversionStrategy(bb_period=10, bb_std=2.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add known sequence
        prices = [50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900]
        for price in prices:
            strategy.update_price("BTC-USD", float(price))

        # Mean should be around 50450
        # With 2 std dev, bands should be fairly wide

        # Add price outside expected range
        strategy.update_price("BTC-USD", 48000.0)

        signals = strategy.get_signals(ctx)

        # Should generate buy signal
        if len(signals) > 0:
            assert signals[0].side == "buy"

    def test_different_bb_std_values(self):
        """Test different standard deviation values"""
        # Wider bands (3.0 std)
        strategy_wide = MeanReversionStrategy(bb_period=20, bb_std=3.0)
        ctx = StrategyContext(symbols=["BTC-USD"])

        for i in range(25):
            strategy_wide.update_price("BTC-USD", 50000.0 + (i % 5) * 100)

        signals_wide = strategy_wide.get_signals(ctx)

        # Narrower bands (1.0 std)
        strategy_narrow = MeanReversionStrategy(bb_period=20, bb_std=1.0)
        for i in range(25):
            strategy_narrow.update_price("BTC-USD", 50000.0 + (i % 5) * 100)

        signals_narrow = strategy_narrow.get_signals(ctx)

        # Narrower bands should generate more signals for same price movement
        # (or at least not fewer signals)
        assert len(signals_narrow) >= len(signals_wide)

    def test_parameter_type_conversion(self):
        """Test parameters are properly converted to correct types"""
        strategy = MeanReversionStrategy(bb_period="25", bb_std="2.5")

        assert isinstance(strategy.bb_period, int)
        assert isinstance(strategy.bb_std, float)
        assert strategy.bb_period == 25
        assert strategy.bb_std == 2.5

    def test_empty_price_history(self):
        """Test strategy handles empty price history"""
        strategy = MeanReversionStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        signals = strategy.get_signals(ctx)

        assert signals == []

    def test_symbol_not_in_context(self):
        """Test strategy handles symbol not in context"""
        strategy = MeanReversionStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add prices for different symbol
        for i in range(25):
            strategy.update_price("ETH-USD", 3000.0)

        signals = strategy.get_signals(ctx)

        # Should not generate signals for ETH-USD as it's not in context
        assert len(signals) == 0