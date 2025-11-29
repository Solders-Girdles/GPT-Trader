"""Tests for state recovery in TradingEngine."""

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import (
    EVENT_PRICE_TICK,
    TradingEngine,
)
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    BaselinePerpsStrategy,
    PerpsStrategyConfig,
)
from gpt_trader.orchestration.configuration import BotConfig


@pytest.fixture
def mock_event_store() -> MagicMock:
    """Create a mock event store."""
    store = MagicMock()
    store.get_recent.return_value = []
    return store


@pytest.fixture
def bot_config() -> BotConfig:
    """Create a minimal bot config."""
    return BotConfig(
        symbols=["BTC-PERP", "ETH-PERP"],
        interval=60,
        strategy=PerpsStrategyConfig(),
    )


@pytest.fixture
def context_with_store(bot_config: BotConfig, mock_event_store: MagicMock) -> CoordinatorContext:
    """Create a context with event store."""
    return CoordinatorContext(
        config=bot_config,
        event_store=mock_event_store,
        bot_id="test-bot",
    )


@pytest.fixture
def context_without_store(bot_config: BotConfig) -> CoordinatorContext:
    """Create a context without event store."""
    return CoordinatorContext(
        config=bot_config,
        bot_id="test-bot",
    )


class TestTradingEngineRehydration:
    """Tests for TradingEngine state recovery."""

    def test_rehydrate_from_empty_events(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test rehydration with no events returns 0."""
        mock_event_store.get_recent.return_value = []
        engine = TradingEngine(context_with_store)

        restored = engine._rehydrate_from_events()

        assert restored == 0
        assert len(engine.price_history) == 0

    def test_rehydrate_from_price_ticks(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test rehydration restores price history from price_tick events."""
        mock_event_store.get_recent.return_value = [
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50000.00"}},
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50100.00"}},
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "ETH-PERP", "price": "3000.00"}},
        ]
        engine = TradingEngine(context_with_store)

        restored = engine._rehydrate_from_events()

        assert restored == 3
        assert len(engine.price_history["BTC-PERP"]) == 2
        assert len(engine.price_history["ETH-PERP"]) == 1
        assert engine.price_history["BTC-PERP"][0] == Decimal("50000.00")
        assert engine.price_history["BTC-PERP"][1] == Decimal("50100.00")

    def test_rehydrate_ignores_unknown_symbols(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test rehydration ignores events for symbols not in config."""
        mock_event_store.get_recent.return_value = [
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50000.00"}},
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "UNKNOWN-PERP", "price": "100.00"}},
        ]
        engine = TradingEngine(context_with_store)

        restored = engine._rehydrate_from_events()

        assert restored == 1
        assert "UNKNOWN-PERP" not in engine.price_history
        assert len(engine.price_history["BTC-PERP"]) == 1

    def test_rehydrate_ignores_other_event_types(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test rehydration ignores non-price_tick events."""
        mock_event_store.get_recent.return_value = [
            {"type": "trade", "data": {"symbol": "BTC-PERP", "price": "50000.00"}},
            {"type": "error", "data": {"message": "test error"}},
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50100.00"}},
        ]
        engine = TradingEngine(context_with_store)

        restored = engine._rehydrate_from_events()

        assert restored == 1
        assert len(engine.price_history["BTC-PERP"]) == 1

    def test_rehydrate_bounds_history_to_20(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test rehydration keeps at most 20 prices per symbol."""
        events = [
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": str(50000 + i)}}
            for i in range(30)
        ]
        mock_event_store.get_recent.return_value = events
        engine = TradingEngine(context_with_store)

        restored = engine._rehydrate_from_events()

        assert restored == 30
        # Should only keep last 20
        assert len(engine.price_history["BTC-PERP"]) == 20
        # First price should be 50010 (30 - 20 = 10, so 50000 + 10)
        assert engine.price_history["BTC-PERP"][0] == Decimal("50010")

    def test_rehydrate_without_event_store(self, context_without_store: CoordinatorContext) -> None:
        """Test rehydration with no event store returns 0 gracefully."""
        engine = TradingEngine(context_without_store)

        restored = engine._rehydrate_from_events()

        assert restored == 0

    def test_rehydrate_handles_invalid_price(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test rehydration handles invalid price values gracefully."""
        mock_event_store.get_recent.return_value = [
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50000.00"}},
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "invalid"}},
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50200.00"}},
        ]
        engine = TradingEngine(context_with_store)

        restored = engine._rehydrate_from_events()

        # Should restore 2 valid prices, skip invalid
        assert restored == 2
        assert len(engine.price_history["BTC-PERP"]) == 2

    def test_rehydrate_handles_missing_fields(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test rehydration handles events with missing fields."""
        mock_event_store.get_recent.return_value = [
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP"}},  # missing price
            {"type": EVENT_PRICE_TICK, "data": {"price": "50000.00"}},  # missing symbol
            {"type": EVENT_PRICE_TICK, "data": {}},  # missing both
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50100.00"}},
        ]
        engine = TradingEngine(context_with_store)

        restored = engine._rehydrate_from_events()

        assert restored == 1
        assert len(engine.price_history["BTC-PERP"]) == 1


class TestTradingEngineRecordPriceTick:
    """Tests for TradingEngine price tick recording."""

    def test_record_price_tick_stores_event(
        self, context_with_store: CoordinatorContext, mock_event_store: MagicMock
    ) -> None:
        """Test that price ticks are recorded to event store."""
        engine = TradingEngine(context_with_store)

        engine._record_price_tick("BTC-PERP", Decimal("50000.00"))

        mock_event_store.store.assert_called_once()
        call_args = mock_event_store.store.call_args[0][0]
        assert call_args["type"] == EVENT_PRICE_TICK
        assert call_args["data"]["symbol"] == "BTC-PERP"
        assert call_args["data"]["price"] == "50000.00"
        assert call_args["data"]["bot_id"] == "test-bot"
        assert "timestamp" in call_args["data"]

    def test_record_price_tick_without_store(
        self, context_without_store: CoordinatorContext
    ) -> None:
        """Test that price tick recording is skipped without event store."""
        engine = TradingEngine(context_without_store)

        # Should not raise
        engine._record_price_tick("BTC-PERP", Decimal("50000.00"))


class TestStrategyRehydrate:
    """Tests for strategy rehydrate method."""

    def test_baseline_strategy_rehydrate_returns_zero(self) -> None:
        """Test that BaselinePerpsStrategy.rehydrate returns 0 (stateless)."""
        strategy = BaselinePerpsStrategy()

        events = [
            {"type": EVENT_PRICE_TICK, "data": {"symbol": "BTC-PERP", "price": "50000.00"}},
        ]
        result = strategy.rehydrate(events)

        assert result == 0
