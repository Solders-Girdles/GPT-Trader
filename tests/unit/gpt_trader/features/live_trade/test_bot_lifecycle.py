"""Tests for TradingBot run/stop lifecycle and running state."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gpt_trader.features.live_trade.bot import TradingBot


class TestTradingBotRun:
    """Test TradingBot run method."""

    @pytest.fixture
    def bot(self) -> TradingBot:
        """Create a TradingBot with mocked engine."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 1  # Short interval for testing
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.start_background_tasks = AsyncMock(return_value=[])
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)

        return bot

    @pytest.mark.asyncio
    async def test_run_sets_running_flag(self, bot: TradingBot) -> None:
        """Test that run sets running flag."""
        await bot.run(single_cycle=True)

        # After completion, running should be False
        assert bot.running is False

    @pytest.mark.asyncio
    async def test_run_single_cycle_calls_shutdown(self, bot: TradingBot) -> None:
        """Test that single_cycle mode calls shutdown."""
        await bot.run(single_cycle=True)

        bot.engine.shutdown.assert_called()

    @pytest.mark.asyncio
    async def test_run_starts_background_tasks(self, bot: TradingBot) -> None:
        """Test that run starts background tasks."""
        await bot.run(single_cycle=True)

        bot.engine.start_background_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_cancellation(self, bot: TradingBot) -> None:
        """Test that run handles CancelledError gracefully."""

        # Make start_background_tasks return a task that gets cancelled
        async def cancelled_task() -> None:
            raise asyncio.CancelledError()

        bot.engine.start_background_tasks = AsyncMock(
            return_value=[asyncio.create_task(cancelled_task())]
        )

        # Should not raise
        await bot.run(single_cycle=False)

        # Shutdown should still be called in finally
        bot.engine.shutdown.assert_called()
        assert bot.running is False


class TestTradingBotStop:
    """Test TradingBot stop/shutdown methods."""

    @pytest.fixture
    def bot(self) -> TradingBot:
        """Create a TradingBot with mocked engine."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)
            bot.running = True

        return bot

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, bot: TradingBot) -> None:
        """Test that stop sets running to False."""
        await bot.stop()

        assert bot.running is False

    @pytest.mark.asyncio
    async def test_stop_calls_engine_shutdown(self, bot: TradingBot) -> None:
        """Test that stop calls engine shutdown."""
        await bot.stop()

        bot.engine.shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_is_alias_for_stop(self, bot: TradingBot) -> None:
        """Test that shutdown is an alias for stop."""
        await bot.shutdown()

        assert bot.running is False
        bot.engine.shutdown.assert_called_once()


class TestTradingBotStateManagement:
    """Test TradingBot state management."""

    def test_initial_state(self) -> None:
        """Test initial state of TradingBot."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 60
        mock_container = Mock()

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            mock_engine.return_value = Mock()
            bot = TradingBot(config=config, container=mock_container)

        assert bot.running is False

    @pytest.mark.asyncio
    async def test_running_state_during_execution(self) -> None:
        """Test running state changes during execution."""
        config = Mock()
        config.symbols = ["BTC-PERP-USDC"]
        config.interval = 0.01  # Very short for testing
        mock_container = Mock()

        running_states: list[bool] = []

        async def capture_state() -> list:
            # Capture state immediately after run starts
            await asyncio.sleep(0.001)
            running_states.append(bot.running)
            return []

        with patch("gpt_trader.features.live_trade.bot.TradingEngine") as mock_engine:
            engine = AsyncMock()
            engine.start_background_tasks = capture_state
            engine.shutdown = AsyncMock()
            mock_engine.return_value = engine
            bot = TradingBot(config=config, container=mock_container)

        # Before run
        assert bot.running is False

        await bot.run(single_cycle=True)

        # During run (captured by capture_state)
        assert running_states == [True]

        # After run
        assert bot.running is False
