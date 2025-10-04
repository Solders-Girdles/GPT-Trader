"""
Characterization Tests for PerpsBot

PURPOSE: Freeze current behavior before refactoring
STATUS: Phase 0 - Expand collaboratively
RULE: These tests document WHAT happens, not HOW it should work

⚠️ These tests may be slow, ugly, or use real resources - that's OK.
⚠️ Goal: Catch ANY behavioral change during refactoring.
⚠️ Add assertions as you discover behavior - this is a living test suite.

Reference: docs/architecture/perps_bot_dependencies.md
"""

import asyncio
import pytest
import threading
from decimal import Decimal
from datetime import datetime, UTC
from unittest.mock import Mock, AsyncMock, patch
from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.features.brokerages.core.interfaces import Quote, Balance, Position


@pytest.fixture
def minimal_config():
    """Minimal config that allows PerpsBot to initialize"""
    return BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-USD"],
        update_interval=60,
        mock_broker=True,
    )


@pytest.fixture
def mock_quote():
    """Standard quote response"""
    quote = Mock(spec=Quote)
    quote.last = 50000.0
    quote.last_price = None
    quote.ts = datetime.now(UTC)
    return quote


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotInitialization:
    """Characterize initialization side effects"""

    def test_initialization_creates_all_services(self, monkeypatch, tmp_path, minimal_config):
        """Document: All services must exist after __init__"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Core services (must exist)
        assert hasattr(bot, "strategy_orchestrator")
        assert hasattr(bot, "execution_coordinator")
        assert hasattr(bot, "system_monitor")
        assert hasattr(bot, "runtime_coordinator")
        assert bot.strategy_orchestrator is not None
        assert bot.execution_coordinator is not None
        assert bot.system_monitor is not None
        assert bot.runtime_coordinator is not None

    def test_initialization_creates_accounting_services(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: Accounting services must exist"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot, "account_manager")
        assert hasattr(bot, "account_telemetry")
        assert bot.account_manager is not None
        assert bot.account_telemetry is not None

    def test_initialization_creates_market_monitor(self, monkeypatch, tmp_path, minimal_config):
        """Document: Market monitor must exist"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot, "_market_monitor")
        assert bot._market_monitor is not None

    def test_initialization_creates_runtime_state(self, monkeypatch, tmp_path, minimal_config):
        """Document: All runtime state dicts must be initialized"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # State dictionaries (NOTE: _product_map removed - was dead code)
        assert hasattr(bot, "mark_windows")
        assert hasattr(bot, "last_decisions")
        assert hasattr(bot, "_last_positions")
        assert hasattr(bot, "order_stats")

        # Verify types
        assert isinstance(bot.mark_windows, dict)
        assert isinstance(bot.last_decisions, dict)
        assert isinstance(bot._last_positions, dict)
        assert isinstance(bot.order_stats, dict)

        # Verify initial values
        assert "BTC-USD" in bot.mark_windows
        assert bot.mark_windows["BTC-USD"] == []
        assert bot.order_stats == {"attempted": 0, "successful": 0, "failed": 0}

    def test_initialization_creates_locks(self, monkeypatch, tmp_path, minimal_config):
        """Document: Threading locks must be created"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot, "_mark_lock")
        # RLock is _thread.RLock type, check by name
        assert type(bot._mark_lock).__name__ == "RLock"

    def test_initialization_sets_symbols(self, monkeypatch, tmp_path, minimal_config):
        """Document: Symbols list is extracted from config"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert bot.symbols == ["BTC-USD"]
        assert isinstance(bot.symbols, list)

    # TODO: Add assertion for derivatives_enabled flag
    # TODO: Add assertion for session_guard creation
    # TODO: Add assertion for config_controller creation
    # TODO: Verify registry broker exists
    # TODO: Verify registry risk_manager exists


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotUpdateMarks:
    """Characterize update_marks behavior"""

    @pytest.mark.asyncio
    async def test_update_marks_updates_mark_windows(
        self, monkeypatch, tmp_path, minimal_config, mock_quote
    ):
        """Document: update_marks must append to mark_windows"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.broker.get_quote = Mock(return_value=mock_quote)

        await bot.update_marks()

        assert len(bot.mark_windows["BTC-USD"]) == 1
        assert bot.mark_windows["BTC-USD"][0] == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_update_marks_updates_risk_manager_timestamp(
        self, monkeypatch, tmp_path, minimal_config, mock_quote
    ):
        """Document: update_marks must update risk_manager.last_mark_update"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.broker.get_quote = Mock(return_value=mock_quote)

        await bot.update_marks()

        assert "BTC-USD" in bot.risk_manager.last_mark_update
        assert isinstance(bot.risk_manager.last_mark_update["BTC-USD"], datetime)

    @pytest.mark.asyncio
    async def test_update_marks_continues_after_symbol_error(
        self, monkeypatch, tmp_path, mock_quote
    ):
        """Document: Error on one symbol must not stop processing others"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
            mock_broker=True,
        )
        bot = PerpsBot(config)

        # ETH-USD will fail, others succeed
        def get_quote_side_effect(symbol):
            if symbol == "ETH-USD":
                raise Exception("ETH quote failed")
            return mock_quote

        bot.broker.get_quote = Mock(side_effect=get_quote_side_effect)

        await bot.update_marks()

        # BTC and SOL should still update despite ETH error
        assert len(bot.mark_windows["BTC-USD"]) == 1
        assert len(bot.mark_windows["ETH-USD"]) == 0  # Failed
        assert len(bot.mark_windows["SOL-USD"]) == 1

    @pytest.mark.asyncio
    async def test_update_marks_trims_window(self, monkeypatch, tmp_path):
        """Document: mark_windows must be trimmed to max(long_ma, short_ma) + 5"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            short_ma=10,
            long_ma=30,
            mock_broker=True,
        )
        bot = PerpsBot(config)

        quote = Mock(spec=Quote)
        quote.last = 50000.0
        quote.ts = datetime.now(UTC)
        bot.broker.get_quote = Mock(return_value=quote)

        # Add 50 marks
        for i in range(50):
            await bot.update_marks()

        max_expected = max(config.long_ma, config.short_ma) + 5  # 35
        assert len(bot.mark_windows["BTC-USD"]) == max_expected

    # TODO: Test concurrent update_marks calls (thread safety)
    # TODO: Test update_marks with None quote
    # TODO: Test update_marks with invalid mark price (<=0)
    # TODO: Verify exception handling preserves risk_manager state


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotProperties:
    """Characterize property descriptor behavior"""

    def test_broker_property_raises_when_none(self, monkeypatch, tmp_path, minimal_config):
        """Document: broker property must raise RuntimeError if None"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        # Registry is frozen dataclass, must use with_updates
        bot.registry = bot.registry.with_updates(broker=None)

        with pytest.raises(RuntimeError) as exc_info:
            _ = bot.broker

        assert "Broker is not configured" in str(exc_info.value)

    def test_risk_manager_property_raises_when_none(self, monkeypatch, tmp_path, minimal_config):
        """Document: risk_manager property must raise RuntimeError if None"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        # Registry is frozen dataclass, must use with_updates
        bot.registry = bot.registry.with_updates(risk_manager=None)

        with pytest.raises(RuntimeError) as exc_info:
            _ = bot.risk_manager

        assert "Risk manager is not configured" in str(exc_info.value)

    def test_exec_engine_property_raises_when_none(self, monkeypatch, tmp_path, minimal_config):
        """Document: exec_engine property must raise RuntimeError if None"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot._exec_engine = None

        with pytest.raises(RuntimeError) as exc_info:
            _ = bot.exec_engine

        assert "Execution engine not initialized" in str(exc_info.value)

    # TODO: Test property setters update registry correctly
    # TODO: Verify properties work after builder construction


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotDelegation:
    """Characterize method delegation patterns"""

    @pytest.mark.asyncio
    async def test_process_symbol_delegates_to_strategy_orchestrator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: process_symbol must delegate to strategy_orchestrator"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.strategy_orchestrator.process_symbol = AsyncMock()

        await bot.process_symbol("BTC-USD")

        bot.strategy_orchestrator.process_symbol.assert_called_once_with("BTC-USD", None, None)

    @pytest.mark.asyncio
    async def test_execute_decision_delegates_to_execution_coordinator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: execute_decision must delegate to execution_coordinator"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)
        bot.execution_coordinator.execute_decision = AsyncMock()

        decision = Mock()
        mark = Decimal("50000")
        product = Mock()
        position_state = {}

        await bot.execute_decision("BTC-USD", decision, mark, product, position_state)

        bot.execution_coordinator.execute_decision.assert_called_once()

    # TODO: Test write_health_status delegation
    # TODO: Test is_reduce_only_mode delegation
    # TODO: Test set_reduce_only_mode delegation


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotStreamingLockSharing:
    """Characterize shared lock between update_marks and streaming"""

    def test_mark_lock_is_reentrant_lock(self, monkeypatch, tmp_path, minimal_config):
        """Document: _mark_lock must be RLock for reentrant access"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # RLock is _thread.RLock type, check by name
        assert type(bot._mark_lock).__name__ == "RLock"

    @pytest.mark.asyncio
    async def test_update_mark_window_is_thread_safe(self, monkeypatch, tmp_path, minimal_config):
        """Document: _update_mark_window must use _mark_lock"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Instrument threading.RLock to track acquire calls
        import threading

        acquire_count = {"count": 0}
        original_rlock = threading.RLock

        class InstrumentedRLock:
            """Wrapper that tracks acquire calls to verify lock usage"""

            def __init__(self):
                self._lock = original_rlock()

            def acquire(self, blocking=True, timeout=-1):
                acquire_count["count"] += 1
                if timeout == -1:
                    return self._lock.acquire(blocking)
                return self._lock.acquire(blocking, timeout)

            def release(self):
                return self._lock.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *args):
                self.release()

        # Patch threading.RLock before bot construction
        monkeypatch.setattr(threading, "RLock", InstrumentedRLock)

        bot = PerpsBot(minimal_config)

        # Reset count after initialization (bot.__init__ may acquire lock)
        acquire_count["count"] = 0

        # Run concurrent updates
        def concurrent_update():
            bot._update_mark_window("BTC-USD", Decimal("50000"))

        thread = threading.Thread(target=concurrent_update)
        thread.start()
        bot._update_mark_window("BTC-USD", Decimal("50100"))
        thread.join(timeout=1.0)

        # CRITICAL: Verify lock was actually acquired (fails if lock removed)
        # Without this assertion, GIL makes list appends safe, hiding lock removal
        assert (
            acquire_count["count"] >= 2
        ), f"Lock acquired {acquire_count['count']} times, expected >= 2"

        # Verify no corruption
        assert len(bot.mark_windows["BTC-USD"]) == 2
        assert all(isinstance(m, Decimal) for m in bot.mark_windows["BTC-USD"])
        assert Decimal("50000") in bot.mark_windows["BTC-USD"]
        assert Decimal("50100") in bot.mark_windows["BTC-USD"]

    def test_streaming_service_shares_mark_lock(self, monkeypatch, tmp_path, minimal_config):
        """Document: StreamingService must use same _mark_lock as PerpsBot"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("USE_NEW_STREAMING_SERVICE", "true")

        bot = PerpsBot(minimal_config)

        # Verify lock sharing via MarketDataService
        if bot._streaming_service is not None:
            assert bot._streaming_service.market_data_service._mark_lock is bot._mark_lock

    # TODO: Test concurrent update_mark_window calls don't race
    # TODO: Verify mark trimming is atomic


@pytest.mark.integration
@pytest.mark.characterization
@pytest.mark.slow
class TestPerpsBotFullCycle:
    """Characterize full lifecycle: init → update → cycle → shutdown"""

    @pytest.mark.asyncio
    async def test_full_cycle_smoke(self, monkeypatch, tmp_path):
        """Document: Full cycle must complete without errors"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            dry_run=True,  # No real orders
            mock_broker=True,
        )
        bot = PerpsBot(config)

        # Setup mock responses
        quote = Mock(spec=Quote)
        quote.last = 50000.0
        quote.ts = datetime.now(UTC)
        bot.broker.get_quote = Mock(return_value=quote)
        bot.broker.list_balances = Mock(return_value=[])
        bot.broker.list_positions = Mock(return_value=[])

        # Execute full cycle
        await bot.update_marks()
        await bot.run_cycle()
        await bot.shutdown()

        # Verify state after cycle
        assert len(bot.mark_windows["BTC-USD"]) > 0
        assert bot.running is False

    # TODO: Test background tasks are spawned in non-dry-run mode
    # TODO: Test all background tasks are canceled on shutdown
    # TODO: Verify shutdown doesn't hang
    # TODO: Test trading window checks


# Feature Flag Tests (for rollback safety)


@pytest.mark.integration
@pytest.mark.characterization
class TestFeatureFlagRollback:
    """Verify legacy code paths work when feature flags are disabled"""

    def test_market_data_service_delegation_enabled(self, monkeypatch, tmp_path, minimal_config):
        """Document: With flag=true, update_marks delegates to MarketDataService"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)
        monkeypatch.setenv("USE_NEW_MARKET_DATA_SERVICE", "true")

        bot = PerpsBot(minimal_config)

        # Verify MarketDataService exists and shares state
        assert hasattr(bot, "_market_data_service")
        assert bot._market_data_service is not None
        assert bot._market_data_service.mark_windows is bot.mark_windows
        assert bot._market_data_service._mark_lock is bot._mark_lock

    def test_legacy_market_data_path_works(self, monkeypatch, tmp_path, minimal_config, mock_quote):
        """Document: With flag=false, update_marks uses legacy path"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)
        monkeypatch.setenv("USE_NEW_MARKET_DATA_SERVICE", "false")

        bot = PerpsBot(minimal_config)

        # Verify service not created when flag is off
        assert bot._market_data_service is None

        # Verify legacy path still works
        bot.broker.get_quote = Mock(return_value=mock_quote)

        import asyncio

        asyncio.run(bot.update_marks())

        # Verify legacy behavior preserved
        assert len(bot.mark_windows["BTC-USD"]) == 1
        assert bot.mark_windows["BTC-USD"][0] == Decimal("50000.0")

    def test_streaming_service_delegation_enabled(self, monkeypatch, tmp_path, minimal_config):
        """Document: With flag=true, streaming uses StreamingService"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("USE_NEW_STREAMING_SERVICE", "true")

        bot = PerpsBot(minimal_config)

        # Verify StreamingService exists when flag is on and MarketDataService exists
        assert hasattr(bot, "_streaming_service")
        if bot._market_data_service is not None:
            assert bot._streaming_service is not None
            assert bot._streaming_service.symbols == bot.symbols
            assert bot._streaming_service.broker is bot.broker
            assert bot._streaming_service.market_data_service is bot._market_data_service
            assert bot._streaming_service.risk_manager is bot.risk_manager

    def test_legacy_streaming_path_works(self, monkeypatch, tmp_path, minimal_config):
        """Ensure USE_NEW_STREAMING_SERVICE=false still works"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("USE_NEW_STREAMING_SERVICE", "false")

        bot = PerpsBot(minimal_config)

        # Verify service not created when flag is off
        assert bot._streaming_service is None

        # Verify legacy methods still exist
        assert hasattr(bot, "_start_streaming_background_legacy")
        assert hasattr(bot, "_stop_streaming_background")
        assert hasattr(bot, "_run_stream_loop")

    # Placeholder for Phase 3
    @pytest.mark.skip(reason="Implement in Phase 3 when Builder is introduced")
    def test_legacy_constructor_path_works(self):
        """Ensure direct PerpsBot() construction still works"""
        pass


@pytest.mark.integration
@pytest.mark.characterization
class TestStreamingServiceRestartBehavior:
    """Characterize streaming restart behavior on config changes"""

    def test_restart_streaming_stops_and_restarts_on_level_change(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: Changing perps_stream_level must restart streaming"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("USE_NEW_STREAMING_SERVICE", "true")

        # Start with streaming enabled
        config = BotConfig(
            profile=Profile.CANARY,  # Required for streaming
            symbols=["BTC-USD"],
            mock_broker=True,
            perps_enable_streaming=True,
            perps_stream_level=1,
        )
        bot = PerpsBot(config)

        # Verify initial streaming service exists
        if bot._streaming_service is not None:
            # Simulate config change with level change
            new_config = BotConfig(
                profile=Profile.CANARY,
                symbols=["BTC-USD"],
                mock_broker=True,
                perps_enable_streaming=True,
                perps_stream_level=2,  # Changed
            )
            from bot_v2.orchestration.config_controller import ConfigChange

            change = ConfigChange(updated=new_config, diff={"perps_stream_level": (1, 2)})

            # Apply change (should trigger restart)
            bot.apply_config_change(change)

            # Verify service still exists after restart
            assert bot._streaming_service is not None

    def test_restart_streaming_stops_when_disabled(self, monkeypatch, tmp_path):
        """Document: Disabling streaming must stop the service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("USE_NEW_STREAMING_SERVICE", "true")

        # Start with streaming enabled
        config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-USD"],
            mock_broker=True,
            perps_enable_streaming=True,
        )
        bot = PerpsBot(config)

        # Simulate disabling streaming
        new_config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-USD"],
            mock_broker=True,
            perps_enable_streaming=False,  # Disabled
        )
        from bot_v2.orchestration.config_controller import ConfigChange

        change = ConfigChange(updated=new_config, diff={"perps_enable_streaming": (True, False)})

        # Apply change (should stop streaming)
        bot.apply_config_change(change)

        # Verify service stopped (if it was created)
        if bot._streaming_service is not None:
            assert not bot._streaming_service.is_running()


@pytest.mark.integration
@pytest.mark.characterization
class TestPerpsBotBuilderPattern:
    """Characterize builder-centric construction now that it is the default."""

    def test_constructor_uses_builder_pipeline(self, monkeypatch, tmp_path, minimal_config):
        """Direct construction routes through PerpsBotBuilder.build."""
        from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        build_calls: list[None] = []
        original_build = PerpsBotBuilder.build

        def tracking_build(self):  # type: ignore[override]
            build_calls.append(None)
            return original_build(self)

        monkeypatch.setattr(PerpsBotBuilder, "build", tracking_build)

        bot = PerpsBot(minimal_config)

        assert build_calls, "Expected PerpsBotBuilder.build to be invoked"
        assert bot.bot_id == "perps_bot"
        assert bot.config == minimal_config
        assert hasattr(bot, "strategy_orchestrator")
        assert hasattr(bot, "execution_coordinator")

    def test_constructor_ignores_legacy_env_flag(self, monkeypatch, tmp_path, minimal_config):
        """Historical USE_PERPS_BOT_BUILDER flag no longer flips code paths."""
        import warnings
        from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setenv("USE_PERPS_BOT_BUILDER", "false")
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        build_calls: list[None] = []
        original_build = PerpsBotBuilder.build

        def tracking_build(self):  # type: ignore[override]
            build_calls.append(None)
            return original_build(self)

        monkeypatch.setattr(PerpsBotBuilder, "build", tracking_build)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            bot = PerpsBot(minimal_config)

        legacy_warnings = [
            w
            for w in caught
            if issubclass(w.category, DeprecationWarning) and "legacy" in str(w.message).lower()
        ]

        assert build_calls, "Builder path should run even when flag is false"
        assert not legacy_warnings, "Legacy builder warnings should be gone"
        assert bot.config == minimal_config

    def test_constructor_and_builder_produce_identical_state(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Direct construction and explicit builder flow return equivalent bots."""
        from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        direct_bot = PerpsBot(minimal_config)
        builder_bot = PerpsBot.from_builder(PerpsBotBuilder(minimal_config))

        core_attrs = [
            "bot_id",
            "running",
            "symbols",
            "_derivatives_enabled",
        ]

        for attr in core_attrs:
            assert getattr(direct_bot, attr) == getattr(builder_bot, attr), attr

        service_attrs = [
            "strategy_orchestrator",
            "execution_coordinator",
            "system_monitor",
            "runtime_coordinator",
            "account_manager",
            "account_telemetry",
            "_market_monitor",
            "event_store",
            "orders_store",
            "config_controller",
            "registry",
        ]

        for attr in service_attrs:
            assert getattr(direct_bot, attr) is not None, f"direct missing {attr}"
            assert getattr(builder_bot, attr) is not None, f"builder missing {attr}"

    def test_builder_from_classmethod_works(self, monkeypatch, tmp_path, minimal_config):
        """PerpsBot.from_builder() creates valid instance"""
        from bot_v2.orchestration.perps_bot_builder import PerpsBotBuilder

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))

        builder = PerpsBotBuilder(minimal_config)
        bot = PerpsBot.from_builder(builder)

        assert isinstance(bot, PerpsBot)
        assert bot.bot_id == "perps_bot"
        assert bot.config == minimal_config
        assert hasattr(bot, "strategy_orchestrator")

    def test_builder_respects_custom_registry(self, monkeypatch, tmp_path, minimal_config):
        """Builder uses provided custom registry"""
        from bot_v2.orchestration.service_registry import empty_registry

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        custom_registry = empty_registry(minimal_config)
        bot = PerpsBot(minimal_config, registry=custom_registry)

        # Verify registry was used (config should match)
        assert bot.registry.config == minimal_config


@pytest.mark.integration
@pytest.mark.characterization
class TestStrategyOrchestratorExtractedServices:
    """Characterize StrategyOrchestrator extracted services (Phase 1-4 refactor)"""

    def test_strategy_orchestrator_has_equity_calculator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have EquityCalculator service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        assert hasattr(bot.strategy_orchestrator, "equity_calculator")
        assert bot.strategy_orchestrator.equity_calculator is not None

    def test_strategy_orchestrator_has_risk_gate_validator(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have RiskGateValidator service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Lazy initialization - access property to trigger creation
        validator = bot.strategy_orchestrator.risk_gate_validator
        assert validator is not None

    def test_strategy_orchestrator_has_strategy_registry(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have StrategyRegistry service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Lazy initialization - access property to trigger creation
        registry = bot.strategy_orchestrator.strategy_registry
        assert registry is not None

    def test_strategy_orchestrator_has_strategy_executor(
        self, monkeypatch, tmp_path, minimal_config
    ):
        """Document: StrategyOrchestrator must have StrategyExecutor service"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Lazy initialization - access property to trigger creation
        executor = bot.strategy_orchestrator.strategy_executor
        assert executor is not None

    @pytest.mark.asyncio
    async def test_process_symbol_uses_extracted_services(
        self, monkeypatch, tmp_path, minimal_config, mock_quote
    ):
        """Document: process_symbol must use extracted services for equity, validation, evaluation"""
        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        bot = PerpsBot(minimal_config)

        # Setup test data
        bot.mark_windows["BTC-USD"] = [Decimal("50000"), Decimal("50100")]
        bot.broker.list_balances = Mock(
            return_value=[Mock(asset="USD", total=Decimal("10000"), available=Decimal("10000"))]
        )
        bot.broker.list_positions = Mock(return_value=[])

        # Initialize strategy
        bot.strategy_orchestrator.init_strategy()

        # Mock extracted services to verify they're called
        original_equity_calc = bot.strategy_orchestrator.equity_calculator.calculate
        original_risk_validate = bot.strategy_orchestrator.risk_gate_validator.validate_gates
        original_strategy_eval = bot.strategy_orchestrator.strategy_executor.evaluate
        original_strategy_record = bot.strategy_orchestrator.strategy_executor.record_decision

        bot.strategy_orchestrator.equity_calculator.calculate = Mock(
            side_effect=original_equity_calc
        )
        bot.strategy_orchestrator.risk_gate_validator.validate_gates = Mock(
            side_effect=original_risk_validate
        )
        bot.strategy_orchestrator.strategy_executor.evaluate = Mock(
            side_effect=original_strategy_eval
        )
        bot.strategy_orchestrator.strategy_executor.record_decision = Mock(
            side_effect=original_strategy_record
        )

        # Execute
        await bot.strategy_orchestrator.process_symbol("BTC-USD")

        # Verify extracted services were used
        bot.strategy_orchestrator.equity_calculator.calculate.assert_called_once()
        bot.strategy_orchestrator.risk_gate_validator.validate_gates.assert_called_once()
        bot.strategy_orchestrator.strategy_executor.evaluate.assert_called_once()
        bot.strategy_orchestrator.strategy_executor.record_decision.assert_called_once()


# TODO: Add more tests collaboratively
# - Product caching (get_product)
# - Error handling edge cases
# - Telemetry side effects
