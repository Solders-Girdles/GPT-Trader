"""Integration tests for crash recovery and state rehydration.

TradingEngine rehydrates price history from EventStore price_tick events after restart.
"""

import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gpt_trader.app import container as app_container
from gpt_trader.app.config import BotConfig
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.persistence.event_store import EventStore

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_container():
    """Set up application container for TradingEngine tests."""
    config = BotConfig(symbols=["BTC-USD"], interval=1, mock_broker=True)
    container = app_container.ApplicationContainer(config)
    app_container.set_application_container(container)
    yield
    app_container.clear_application_container()


def create_test_config(symbols: list[str] | None = None) -> BotConfig:
    """Create a minimal BotConfig for testing."""
    return BotConfig(
        symbols=symbols or ["BTC-USD"],
        interval=1,
        mock_broker=True,
    )


def create_test_context(
    config: BotConfig, event_store: EventStore | None = None
) -> CoordinatorContext:
    """Create a CoordinatorContext for testing."""
    container = app_container.get_application_container()
    return CoordinatorContext(
        config=config,
        container=container,
        broker=MagicMock(),
        symbols=tuple(config.symbols),
        event_store=event_store,
    )


class TestCrashRecovery:
    """Test state recovery after simulated crashes."""

    def test_price_history_survives_restart(self) -> None:
        """Price history should be restored from EventStore after restart."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = create_test_config(symbols=["BTC-USD", "ETH-USD"])

            # === Session 1: Record price ticks ===
            with EventStore(root=root) as store1:
                context1 = create_test_context(config, event_store=store1)
                engine1 = TradingEngine(context1)

                # Simulate price ticks being recorded
                btc_prices = [Decimal("50000"), Decimal("50100"), Decimal("50200")]
                eth_prices = [Decimal("3000"), Decimal("3050")]

                for price in btc_prices:
                    # _record_price_tick adds to both EventStore and price_history
                    engine1._record_price_tick("BTC-USD", price)

                for price in eth_prices:
                    engine1._record_price_tick("ETH-USD", price)

                # Verify engine1 has the prices
                assert len(engine1.price_history["BTC-USD"]) == 3
                assert len(engine1.price_history["ETH-USD"]) == 2

                # Simulate crash - destroy engine1
                del engine1

            # === Session 2: Recover from EventStore ===
            with EventStore(root=root) as store2:
                context2 = create_test_context(config, event_store=store2)
                engine2 = TradingEngine(context2)

                # Engine should start with empty price_history
                assert len(engine2.price_history["BTC-USD"]) == 0

                # Rehydrate from events (this is called in start_background_tasks)
                restored_count = engine2._rehydrate_from_events()

                # Verify prices were restored
                assert restored_count == 5  # 3 BTC + 2 ETH
                assert len(engine2.price_history["BTC-USD"]) == 3
                assert len(engine2.price_history["ETH-USD"]) == 2

                # Verify price values match
                assert engine2.price_history["BTC-USD"][0] == Decimal("50000")
                assert engine2.price_history["BTC-USD"][2] == Decimal("50200")
                assert engine2.price_history["ETH-USD"][0] == Decimal("3000")

    def test_rehydration_filters_by_configured_symbols(self) -> None:
        """Rehydration should only restore prices for symbols in config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            # Session 1: Record prices for multiple symbols
            with EventStore(root=root) as store1:
                config1 = create_test_config(symbols=["BTC-USD", "ETH-USD", "SOL-USD"])
                context1 = create_test_context(config1, event_store=store1)
                engine1 = TradingEngine(context1)

                engine1._record_price_tick("BTC-USD", Decimal("50000"))
                engine1._record_price_tick("ETH-USD", Decimal("3000"))
                engine1._record_price_tick("SOL-USD", Decimal("100"))

            # Session 2: Only configure BTC-USD
            with EventStore(root=root) as store2:
                config2 = create_test_config(symbols=["BTC-USD"])  # Only BTC
                context2 = create_test_context(config2, event_store=store2)
                engine2 = TradingEngine(context2)

                restored_count = engine2._rehydrate_from_events()

                # Should only restore BTC-USD
                assert restored_count == 1
                assert len(engine2.price_history["BTC-USD"]) == 1
                assert len(engine2.price_history["ETH-USD"]) == 0  # Not configured
                assert len(engine2.price_history["SOL-USD"]) == 0  # Not configured

    def test_rehydration_respects_max_history_size(self) -> None:
        """Rehydration should keep price history bounded to 20 items."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = create_test_config(symbols=["BTC-USD"])

            # Session 1: Record 30 prices (more than max 20)
            with EventStore(root=root) as store1:
                context1 = create_test_context(config, event_store=store1)
                engine1 = TradingEngine(context1)

                for i in range(30):
                    price = Decimal(f"{50000 + i}")
                    engine1._record_price_tick("BTC-USD", price)

            # Session 2: Rehydrate
            with EventStore(root=root) as store2:
                context2 = create_test_context(config, event_store=store2)
                engine2 = TradingEngine(context2)

                engine2._rehydrate_from_events()

                # Should only have last 20 prices
                assert len(engine2.price_history["BTC-USD"]) == 20
                # First price should be the 11th (index 10, value 50010)
                assert engine2.price_history["BTC-USD"][0] == Decimal("50010")
                # Last price should be 50029
                assert engine2.price_history["BTC-USD"][-1] == Decimal("50029")

    def test_in_memory_mode_no_persistence(self) -> None:
        """In-memory EventStore should not persist across restarts."""
        config = create_test_config(symbols=["BTC-USD"])

        # Session 1: In-memory store
        store1 = EventStore()  # No root = in-memory
        context1 = create_test_context(config, event_store=store1)
        engine1 = TradingEngine(context1)

        engine1._record_price_tick("BTC-USD", Decimal("50000"))

        # Verify event was recorded in memory
        assert len(store1.list_events()) == 1

        # Session 2: New in-memory store
        store2 = EventStore()  # New in-memory store
        context2 = create_test_context(config, event_store=store2)
        engine2 = TradingEngine(context2)

        restored = engine2._rehydrate_from_events()

        # Nothing to restore from empty in-memory store
        assert restored == 0
        assert len(engine2.price_history["BTC-USD"]) == 0

    def test_no_event_store_graceful_degradation(self) -> None:
        """Engine should work without EventStore (no persistence)."""
        config = create_test_config(symbols=["BTC-USD"])
        context = create_test_context(config, event_store=None)
        engine = TradingEngine(context)

        # Record should be no-op without event store
        engine._record_price_tick("BTC-USD", Decimal("50000"))

        # Rehydrate should return 0 without event store
        restored = engine._rehydrate_from_events()
        assert restored == 0

    def test_multiple_restarts_accumulate_correctly(self) -> None:
        """Multiple restart cycles should accumulate prices correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = create_test_config(symbols=["BTC-USD"])

            # Session 1: Record 5 prices
            with EventStore(root=root) as store1:
                context1 = create_test_context(config, event_store=store1)
                engine1 = TradingEngine(context1)

                for i in range(5):
                    engine1._record_price_tick("BTC-USD", Decimal(f"{50000 + i}"))

            # Session 2: Rehydrate and add 5 more
            with EventStore(root=root) as store2:
                context2 = create_test_context(config, event_store=store2)
                engine2 = TradingEngine(context2)
                engine2._rehydrate_from_events()

                # Add 5 more prices
                for i in range(5, 10):
                    engine2._record_price_tick("BTC-USD", Decimal(f"{50000 + i}"))

            # Session 3: Should see all 10 prices
            with EventStore(root=root) as store3:
                context3 = create_test_context(config, event_store=store3)
                engine3 = TradingEngine(context3)
                restored = engine3._rehydrate_from_events()

                assert restored == 10
                assert len(engine3.price_history["BTC-USD"]) == 10
                assert engine3.price_history["BTC-USD"][0] == Decimal("50000")
                assert engine3.price_history["BTC-USD"][-1] == Decimal("50009")
