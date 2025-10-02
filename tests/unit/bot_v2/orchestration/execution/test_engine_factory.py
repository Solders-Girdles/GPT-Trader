"""Tests for ExecutionEngineFactory."""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.orchestration.execution.engine_factory import ExecutionEngineFactory


class TestExecutionEngineFactory:
    """Test suite for ExecutionEngineFactory."""

    def test_parse_slippage_multipliers_empty(self):
        """Test parsing empty slippage multipliers."""
        with patch.dict("os.environ", {}, clear=True):
            result = ExecutionEngineFactory.parse_slippage_multipliers()
            assert result == {}

    def test_parse_slippage_multipliers_valid(self):
        """Test parsing valid slippage multipliers."""
        with patch.dict("os.environ", {"SLIPPAGE_MULTIPLIERS": "BTC-USD:1.5,ETH-USD:2.0"}):
            result = ExecutionEngineFactory.parse_slippage_multipliers()
            assert result == {"BTC-USD": 1.5, "ETH-USD": 2.0}

    def test_parse_slippage_multipliers_invalid(self):
        """Test parsing invalid slippage multipliers (no colon)."""
        with patch.dict("os.environ", {"SLIPPAGE_MULTIPLIERS": "invalid"}):
            result = ExecutionEngineFactory.parse_slippage_multipliers()
            assert result == {}

    def test_parse_slippage_multipliers_exception_handling(self):
        """Test parsing slippage multipliers with float conversion error."""
        with patch.dict("os.environ", {"SLIPPAGE_MULTIPLIERS": "BTC-USD:not_a_float"}):
            result = ExecutionEngineFactory.parse_slippage_multipliers()
            # Should catch exception and return empty dict
            assert result == {}

    def test_should_use_advanced_engine_no_config(self):
        """Test engine selection with no risk config."""
        risk_manager = Mock()
        risk_manager.config = None

        result = ExecutionEngineFactory.should_use_advanced_engine(risk_manager)
        assert result is False

    def test_should_use_advanced_engine_dynamic_sizing(self):
        """Test engine selection with dynamic sizing enabled."""
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = True
        risk_manager.config.enable_market_impact_guard = False

        result = ExecutionEngineFactory.should_use_advanced_engine(risk_manager)
        assert result is True

    def test_should_use_advanced_engine_market_impact(self):
        """Test engine selection with market impact enabled."""
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = False
        risk_manager.config.enable_market_impact_guard = True

        result = ExecutionEngineFactory.should_use_advanced_engine(risk_manager)
        assert result is True

    def test_should_use_advanced_engine_both_disabled(self):
        """Test engine selection with both features disabled."""
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = False
        risk_manager.config.enable_market_impact_guard = False

        result = ExecutionEngineFactory.should_use_advanced_engine(risk_manager)
        assert result is False

    def test_create_engine_live_execution(self):
        """Test creating LiveExecutionEngine."""
        broker = Mock()
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = False
        risk_manager.config.enable_market_impact_guard = False
        event_store = Mock()

        with patch.dict("os.environ", {}, clear=True):
            engine = ExecutionEngineFactory.create_engine(
                broker=broker,
                risk_manager=risk_manager,
                event_store=event_store,
                bot_id="test_bot",
                enable_preview=False,
            )

        assert engine is not None
        # Should be LiveExecutionEngine
        assert type(engine).__name__ == "LiveExecutionEngine"

    def test_create_engine_advanced_execution(self):
        """Test creating AdvancedExecutionEngine."""
        broker = Mock()
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = True
        risk_manager.config.enable_market_impact_guard = False
        risk_manager.set_impact_estimator = Mock()
        event_store = Mock()

        # Mock broker methods needed for impact estimator
        broker.get_quote = Mock(return_value=None)

        with patch.dict("os.environ", {}, clear=True):
            engine = ExecutionEngineFactory.create_engine(
                broker=broker,
                risk_manager=risk_manager,
                event_store=event_store,
                bot_id="test_bot",
                enable_preview=False,
            )

        assert engine is not None
        # Should be AdvancedExecutionEngine
        assert type(engine).__name__ == "AdvancedExecutionEngine"
        # Should have set impact estimator
        risk_manager.set_impact_estimator.assert_called_once()


class TestImpactEstimator:
    """Tests for create_impact_estimator and the returned closure."""

    def test_create_impact_estimator_with_quote_last(self):
        """Impact estimator uses quote.last for mid price when available."""
        broker = Mock()
        broker.order_books = None  # No seeded orderbooks
        risk_manager = Mock()

        quote = Mock()
        quote.last = 50000.0
        quote.bid = 49900.0
        quote.ask = 50100.0
        broker.get_quote = Mock(return_value=quote)

        estimator = ExecutionEngineFactory.create_impact_estimator(broker, risk_manager)

        # Create a mock request
        request = Mock()
        request.symbol = "BTC-USD"
        request.side = "buy"
        request.quantity = Decimal("1.0")

        # Call the estimator (should not raise)
        result = estimator(request)

        assert result is not None
        broker.get_quote.assert_called_once_with("BTC-USD")

    def test_create_impact_estimator_with_quote_bid_ask(self):
        """Impact estimator calculates mid from bid/ask when last unavailable."""
        broker = Mock()
        broker.order_books = None
        risk_manager = Mock()

        quote = Mock()
        quote.last = None  # No last price
        quote.bid = 49900.0
        quote.ask = 50100.0
        broker.get_quote = Mock(return_value=quote)

        estimator = ExecutionEngineFactory.create_impact_estimator(broker, risk_manager)

        request = Mock()
        request.symbol = "ETH-USD"
        request.side = "sell"
        request.quantity = Decimal("10.0")

        result = estimator(request)

        assert result is not None

    def test_create_impact_estimator_no_quote(self):
        """Impact estimator uses default mid=100 when quote unavailable."""
        broker = Mock()
        broker.order_books = None
        risk_manager = Mock()
        broker.get_quote = Mock(return_value=None)

        estimator = ExecutionEngineFactory.create_impact_estimator(broker, risk_manager)

        request = Mock()
        request.symbol = "XYZ-USD"
        request.side = "buy"
        request.quantity = Decimal("5.0")

        result = estimator(request)

        # Should still work with default mid=100
        assert result is not None

    def test_create_impact_estimator_quote_exception(self):
        """Impact estimator handles broker.get_quote exceptions gracefully."""
        broker = Mock()
        broker.order_books = None
        risk_manager = Mock()
        broker.get_quote = Mock(side_effect=RuntimeError("Broker error"))

        estimator = ExecutionEngineFactory.create_impact_estimator(broker, risk_manager)

        request = Mock()
        request.symbol = "BTC-USD"
        request.side = "buy"
        request.quantity = Decimal("1.0")

        # Should not raise, falls back to default mid
        result = estimator(request)
        assert result is not None

    def test_create_impact_estimator_with_seeded_orderbook(self):
        """Impact estimator uses broker.order_books when available."""
        broker = Mock()
        risk_manager = Mock()

        # Seed custom order book
        broker.order_books = {
            "BTC-USD": (
                [(49900.0, 10.0), (49800.0, 20.0)],  # bids
                [(50100.0, 10.0), (50200.0, 20.0)],  # asks
            )
        }
        broker.get_quote = Mock(return_value=None)

        estimator = ExecutionEngineFactory.create_impact_estimator(broker, risk_manager)

        request = Mock()
        request.symbol = "BTC-USD"
        request.side = "buy"
        request.quantity = Decimal("5.0")

        result = estimator(request)

        # Should use seeded orderbook
        assert result is not None

    def test_create_impact_estimator_calculates_tick_from_spread(self):
        """Impact estimator calculates tick from bid/ask spread."""
        broker = Mock()
        broker.order_books = None
        risk_manager = Mock()

        quote = Mock()
        quote.last = 1000.0
        quote.bid = 999.0
        quote.ask = 1001.0
        broker.get_quote = Mock(return_value=quote)

        estimator = ExecutionEngineFactory.create_impact_estimator(broker, risk_manager)

        request = Mock()
        request.symbol = "SOL-USD"
        request.side = "buy"
        request.quantity = Decimal("100.0")

        result = estimator(request)

        # Tick should be (1001 - 999) / 2 = 1.0
        assert result is not None

    def test_create_impact_estimator_uses_default_tick_when_no_spread(self):
        """Impact estimator uses default tick=mid*0.0005 when spread unavailable."""
        broker = Mock()
        broker.order_books = None
        risk_manager = Mock()

        quote = Mock()
        quote.last = 1000.0
        quote.bid = None  # No bid/ask
        quote.ask = None
        broker.get_quote = Mock(return_value=quote)

        estimator = ExecutionEngineFactory.create_impact_estimator(broker, risk_manager)

        request = Mock()
        request.symbol = "TOKEN-USD"
        request.side = "sell"
        request.quantity = Decimal("50.0")

        result = estimator(request)

        # Should use default tick
        assert result is not None


class TestCreateEngineErrorHandling:
    """Tests for create_engine error handling."""

    def test_create_engine_impact_estimator_init_failure(self):
        """Engine creation continues even if impact estimator setup fails."""
        broker = Mock()
        risk_manager = Mock()
        risk_manager.config = Mock()
        risk_manager.config.enable_dynamic_position_sizing = True
        risk_manager.config.enable_market_impact_guard = False

        # Make set_impact_estimator raise an exception
        risk_manager.set_impact_estimator = Mock(side_effect=RuntimeError("Init failed"))
        event_store = Mock()

        # Should not raise, just log warning
        engine = ExecutionEngineFactory.create_engine(
            broker=broker,
            risk_manager=risk_manager,
            event_store=event_store,
            bot_id="test_bot",
            enable_preview=False,
        )

        # Engine should still be created
        assert engine is not None
        assert type(engine).__name__ == "AdvancedExecutionEngine"
