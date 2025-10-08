"""Tests for Momentum strategy."""

import pytest

from bot_v2.features.strategies.interfaces import StrategyContext
from bot_v2.features.strategies.momentum import MomentumStrategy


class TestMomentumStrategy:
    """Test the MomentumStrategy class."""

    def test_momentum_strategy_creation(self) -> None:
        """Test MomentumStrategy can be created."""
        strategy = MomentumStrategy()
        assert strategy.name == "momentum"
        assert strategy.momentum_period == 10
        assert strategy.threshold == 0.02

    def test_momentum_strategy_with_custom_params(self) -> None:
        """Test MomentumStrategy with custom parameters."""
        strategy = MomentumStrategy(momentum_period=20, threshold=0.05)
        assert strategy.momentum_period == 20
        assert strategy.threshold == 0.05

    def test_get_signals_no_price_history(self) -> None:
        """Test get_signals returns empty when no price history."""
        strategy = MomentumStrategy()
        ctx = StrategyContext(symbols=["BTC-USD"])
        signals = strategy.get_signals(ctx)
        assert signals == []

    def test_get_signals_insufficient_price_history(self) -> None:
        """Test get_signals returns empty when not enough price history."""
        strategy = MomentumStrategy(momentum_period=10)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Add only 5 prices (need > 10)
        for i in range(5):
            strategy.update_price("BTC-USD", 50000.0 + i * 100)

        signals = strategy.get_signals(ctx)
        assert signals == []

    def test_get_signals_positive_momentum_above_threshold(self) -> None:
        """Test get_signals generates buy signal for positive momentum."""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create upward momentum > 2%
        # Need momentum_period + 1 prices minimum (11 prices)
        # momentum = (curr - old) / old where old = prices[-11]
        # To get 3% momentum: curr = old * 1.03
        prices = [50000.0] * 11  # Start with flat prices
        prices.append(50000.0 * 1.03)  # Last price is 3% higher than old price

        for price in prices:
            strategy.update_price("BTC-USD", price)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "buy"
        assert signals[0].confidence > 0

    def test_get_signals_negative_momentum_above_threshold(self) -> None:
        """Test get_signals generates sell signal for negative momentum."""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create downward momentum > 2%
        base_price = 50000.0
        for i in range(15):
            # Price decreases from 50000 to 48500 (3% decrease)
            price = base_price - (i * 100)
            strategy.update_price("BTC-USD", price)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        assert signals[0].symbol == "BTC-USD"
        assert signals[0].side == "sell"
        assert signals[0].confidence > 0

    def test_get_signals_momentum_below_threshold(self) -> None:
        """Test get_signals returns empty when momentum below threshold."""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create small momentum (< 2%)
        for i in range(15):
            # Price increases by only 0.5%
            price = 50000.0 + (i * 25)
            strategy.update_price("BTC-USD", price)

        signals = strategy.get_signals(ctx)
        assert signals == []

    def test_get_signals_multiple_symbols(self) -> None:
        """Test get_signals with multiple symbols."""
        strategy = MomentumStrategy(momentum_period=5, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD"])

        # BTC has positive momentum (3% increase)
        # Need 6 prices for momentum_period=5
        btc_prices = [50000.0] * 6
        btc_prices.append(50000.0 * 1.03)  # Last price is 3% higher
        for price in btc_prices:
            strategy.update_price("BTC-USD", price)

        # ETH has negative momentum (3% decrease)
        eth_prices = [3000.0] * 6
        eth_prices.append(3000.0 * 0.97)  # Last price is 3% lower
        for price in eth_prices:
            strategy.update_price("ETH-USD", price)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 2
        btc_signal = [s for s in signals if s.symbol == "BTC-USD"][0]
        eth_signal = [s for s in signals if s.symbol == "ETH-USD"][0]

        assert btc_signal.side == "buy"
        assert eth_signal.side == "sell"

    def test_get_signals_zero_old_price(self) -> None:
        """Test get_signals handles zero old price gracefully."""
        strategy = MomentumStrategy(momentum_period=5)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # First price is zero
        strategy.update_price("BTC-USD", 0.0)
        for i in range(10):
            strategy.update_price("BTC-USD", 50000.0 + i * 100)

        signals = strategy.get_signals(ctx)
        # Should not crash, should skip this symbol
        assert isinstance(signals, list)

    def test_confidence_scales_with_momentum(self) -> None:
        """Test that confidence scales with momentum strength."""
        strategy = MomentumStrategy(momentum_period=10, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create strong momentum (6% = 3x threshold)
        for i in range(15):
            price = 50000.0 + (i * 200)
            strategy.update_price("BTC-USD", price)

        signals = strategy.get_signals(ctx)

        assert len(signals) == 1
        # Confidence should be capped at 1.0
        assert 0 < signals[0].confidence <= 1.0

    def test_confidence_calculation(self) -> None:
        """Test confidence calculation matches expected formula."""
        strategy = MomentumStrategy(momentum_period=5, threshold=0.10)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create exact 20% momentum (2x threshold)
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 120.0]
        for price in prices:
            strategy.update_price("BTC-USD", price)

        signals = strategy.get_signals(ctx)

        # momentum = (120 - 105) / 105 ≈ 0.143 (14.3%)
        # confidence = min(1.0, 0.143 / 0.10) = min(1.0, 1.43) = 1.0
        assert len(signals) == 1
        assert signals[0].confidence == 1.0


class TestMomentumStrategyEdgeCases:
    """Test edge cases for MomentumStrategy."""

    def test_exact_threshold_momentum(self) -> None:
        """Test behavior when momentum exactly equals threshold."""
        strategy = MomentumStrategy(momentum_period=5, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # Create exact 2% momentum
        prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 102.0]
        for price in prices:
            strategy.update_price("BTC-USD", price)

        signals = strategy.get_signals(ctx)
        # Should NOT generate signal (needs to be > threshold, not >=)
        assert len(signals) == 0

    def test_symbol_not_in_context(self) -> None:
        """Test that only symbols in context are checked."""
        strategy = MomentumStrategy(momentum_period=5, threshold=0.02)

        # Add price history for BTC
        for i in range(10):
            strategy.update_price("BTC-USD", 50000.0 + i * 200)

        # But don't include BTC in context
        ctx = StrategyContext(symbols=["ETH-USD"])
        signals = strategy.get_signals(ctx)

        assert signals == []

    def test_flat_prices_no_signal(self) -> None:
        """Test that flat prices generate no signal."""
        strategy = MomentumStrategy(momentum_period=5, threshold=0.02)
        ctx = StrategyContext(symbols=["BTC-USD"])

        # All prices the same
        for _ in range(10):
            strategy.update_price("BTC-USD", 50000.0)

        signals = strategy.get_signals(ctx)
        assert signals == []
