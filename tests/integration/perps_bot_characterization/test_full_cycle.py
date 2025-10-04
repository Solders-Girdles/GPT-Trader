"""
Characterization Tests for PerpsBot Full Lifecycle

Tests documenting full lifecycle: init → update → cycle → shutdown.
"""

import pytest
from datetime import datetime, UTC
from unittest.mock import Mock

from bot_v2.orchestration.perps_bot import PerpsBot
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.features.brokerages.core.interfaces import Quote


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

    @pytest.mark.asyncio
    async def test_background_tasks_spawned(self, monkeypatch, tmp_path):
        """Document: Background tasks must be spawned in non-dry-run mode"""
        from bot_v2.orchestration.configuration import BotConfig, Profile

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Non-dry-run config to trigger background tasks
        config = BotConfig(
            profile=Profile.CANARY,  # Non-dev profile
            symbols=["BTC-USD"],
            mock_broker=True,
            dry_run=False,  # Explicit non-dry-run
        )
        bot = PerpsBot(config)

        # Configure background tasks (what run() does before starting)
        bot.lifecycle_service.configure_background_tasks(single_cycle=False)

        # Verify tasks were registered (not spawned yet, but configured)
        # LifecycleService should have task factories registered
        assert len(bot.lifecycle_service._task_registry._factory_functions) > 0

        # Expected tasks:
        # 1. Runtime guards
        # 2. Order reconciliation
        # 3. Position reconciliation
        # 4. Account telemetry (maybe - depends on broker support)
        # 5. Execution metrics export
        # Minimum 3 tasks guaranteed (guards, order reconciliation, position reconciliation)
        assert len(bot.lifecycle_service._task_registry._factory_functions) >= 3

    @pytest.mark.asyncio
    async def test_background_tasks_canceled_on_shutdown(self, monkeypatch, tmp_path):
        """Document: Background tasks must be canceled during cleanup"""
        from bot_v2.orchestration.configuration import BotConfig, Profile
        import asyncio

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.CANARY,
            symbols=["BTC-USD"],
            mock_broker=True,
            dry_run=False,
        )
        bot = PerpsBot(config)

        # Create mock background tasks
        async def mock_task():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                raise  # Proper cancellation

        # Spawn some tasks via registry
        task1 = asyncio.create_task(mock_task())
        task2 = asyncio.create_task(mock_task())
        bot.lifecycle_service._task_registry._tasks = [task1, task2]

        # Call cleanup (what shutdown does)
        await bot.lifecycle_service._cleanup()

        # Verify tasks were canceled
        assert task1.cancelled() or task1.done()
        assert task2.cancelled() or task2.done()

    @pytest.mark.asyncio
    async def test_shutdown_doesnt_hang(self, monkeypatch, tmp_path):
        """Document: Shutdown must complete within reasonable timeout"""
        from bot_v2.orchestration.configuration import BotConfig, Profile
        import asyncio

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            mock_broker=True,
        )
        bot = PerpsBot(config)

        # Shutdown should complete quickly (no background tasks to wait for)
        start = asyncio.get_event_loop().time()
        await bot.shutdown()
        elapsed = asyncio.get_event_loop().time() - start

        # Shutdown should be nearly instantaneous (< 1 second)
        assert elapsed < 1.0
        assert bot.running is False

    @pytest.mark.asyncio
    async def test_trading_window_checks(self, monkeypatch, tmp_path):
        """Document: run_cycle must skip trading when outside trading window"""
        from datetime import time
        from bot_v2.orchestration.configuration import BotConfig, Profile

        monkeypatch.setenv("EVENT_STORE_ROOT", str(tmp_path))
        monkeypatch.setattr(PerpsBot, "_start_streaming_if_configured", lambda self: None)

        # Configure trading window: Mon-Fri, 9:00-17:00
        config = BotConfig(
            profile=Profile.DEV,
            symbols=["BTC-USD"],
            mock_broker=True,
            trading_window_start=time(9, 0),
            trading_window_end=time(17, 0),
            trading_days=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        )
        bot = PerpsBot(config)

        # Mock session guard to return False (outside window)
        bot._session_guard.should_trade = lambda: False

        # Mock process_symbol to track if trading logic runs
        from unittest.mock import AsyncMock

        bot.process_symbol = AsyncMock()

        # Run cycle
        await bot.run_cycle()

        # Verify trading logic was skipped
        bot.process_symbol.assert_not_called()
