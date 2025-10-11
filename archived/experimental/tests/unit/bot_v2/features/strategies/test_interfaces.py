"""Tests for strategy interfaces and base classes."""

import pytest

from bot_v2.features.strategies.interfaces import (
    StrategyBase,
    StrategyContext,
    StrategySignal,
)


class TestStrategySignal:
    """Test the StrategySignal dataclass."""

    def test_strategy_signal_creation(self) -> None:
        """Test StrategySignal can be created with required fields."""
        signal = StrategySignal(symbol="BTC-USD", side="buy")
        assert signal.symbol == "BTC-USD"
        assert signal.side == "buy"
        assert signal.confidence == 1.0

    def test_strategy_signal_with_confidence(self) -> None:
        """Test StrategySignal with custom confidence."""
        signal = StrategySignal(symbol="ETH-USD", side="sell", confidence=0.75)
        assert signal.symbol == "ETH-USD"
        assert signal.side == "sell"
        assert signal.confidence == 0.75

    def test_strategy_signal_hold(self) -> None:
        """Test StrategySignal with hold side."""
        signal = StrategySignal(symbol="BTC-USD", side="hold", confidence=0.5)
        assert signal.side == "hold"
        assert signal.confidence == 0.5


class TestStrategyContext:
    """Test the StrategyContext dataclass."""

    def test_strategy_context_creation(self) -> None:
        """Test StrategyContext can be created."""
        ctx = StrategyContext(symbols=["BTC-USD", "ETH-USD"])
        assert ctx.symbols == ["BTC-USD", "ETH-USD"]

    def test_strategy_context_empty_symbols(self) -> None:
        """Test StrategyContext with empty symbols list."""
        ctx = StrategyContext(symbols=[])
        assert ctx.symbols == []


class TestStrategyBase:
    """Test the StrategyBase class."""

    def test_strategy_base_creation(self) -> None:
        """Test StrategyBase can be created."""
        strategy = StrategyBase()
        assert strategy.name == "base"
        assert strategy.params == {}
        assert strategy.price_history == {}

    def test_strategy_base_with_params(self) -> None:
        """Test StrategyBase with custom parameters."""
        strategy = StrategyBase(lookback=100, threshold=0.02)
        assert strategy.params == {"lookback": 100, "threshold": 0.02}

    def test_update_price_adds_to_history(self) -> None:
        """Test that update_price adds price to history."""
        strategy = StrategyBase()
        strategy.update_price("BTC-USD", 50000.0)
        strategy.update_price("BTC-USD", 51000.0)

        assert "BTC-USD" in strategy.price_history
        assert strategy.price_history["BTC-USD"] == [50000.0, 51000.0]

    def test_update_price_multiple_symbols(self) -> None:
        """Test update_price with multiple symbols."""
        strategy = StrategyBase()
        strategy.update_price("BTC-USD", 50000.0)
        strategy.update_price("ETH-USD", 3000.0)
        strategy.update_price("BTC-USD", 51000.0)

        assert len(strategy.price_history) == 2
        assert strategy.price_history["BTC-USD"] == [50000.0, 51000.0]
        assert strategy.price_history["ETH-USD"] == [3000.0]

    def test_update_price_respects_lookback(self) -> None:
        """Test that price history is capped at lookback length."""
        strategy = StrategyBase(lookback=3)

        # Add more than lookback prices
        for price in [100, 200, 300, 400, 500]:
            strategy.update_price("BTC-USD", float(price))

        # Should only keep last 3
        assert len(strategy.price_history["BTC-USD"]) == 3
        assert strategy.price_history["BTC-USD"] == [300.0, 400.0, 500.0]

    def test_update_price_default_lookback(self) -> None:
        """Test that default lookback is 500."""
        strategy = StrategyBase()

        # Add 600 prices
        for i in range(600):
            strategy.update_price("BTC-USD", float(i))

        # Should be capped at 500
        assert len(strategy.price_history["BTC-USD"]) == 500
        # Should have the last 500 (from 100 to 599)
        assert strategy.price_history["BTC-USD"][0] == 100.0
        assert strategy.price_history["BTC-USD"][-1] == 599.0

    def test_get_signals_returns_empty_list(self) -> None:
        """Test that base class get_signals returns empty list."""
        strategy = StrategyBase()
        ctx = StrategyContext(symbols=["BTC-USD"])
        signals = strategy.get_signals(ctx)
        assert signals == []

    def test_name_attribute(self) -> None:
        """Test that strategy has name attribute."""
        strategy = StrategyBase()
        assert hasattr(strategy, "name")
        assert strategy.name == "base"


class TestStrategyProtocol:
    """Test that concrete strategies conform to the protocol."""

    def test_strategy_base_has_required_methods(self) -> None:
        """Test that StrategyBase has all required methods."""
        strategy = StrategyBase()

        # Should have name attribute
        assert hasattr(strategy, "name")

        # Should have update_price method
        assert hasattr(strategy, "update_price")
        assert callable(strategy.update_price)

        # Should have get_signals method
        assert hasattr(strategy, "get_signals")
        assert callable(strategy.get_signals)

    def test_strategy_base_method_signatures(self) -> None:
        """Test that StrategyBase methods have correct signatures."""
        strategy = StrategyBase()

        # update_price should accept symbol and price
        strategy.update_price("BTC-USD", 50000.0)

        # get_signals should accept context and return list
        ctx = StrategyContext(symbols=[])
        result = strategy.get_signals(ctx)
        assert isinstance(result, list)
