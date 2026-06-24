"""Tests for TUI bot lifecycle manager behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.worker import WorkerState

from gpt_trader.tui.managers.bot_lifecycle import BotLifecycleManager


class PaperBroker:
    pass


class RealBroker:
    pass


def _app_for_start_mode(
    mode: str,
    *,
    live_confirmed: bool = False,
    warning_result: bool = True,
    broker: object | None = None,
) -> MagicMock:
    app = MagicMock()
    app.bot = SimpleNamespace(
        running=False,
        engine=SimpleNamespace(broker=broker or PaperBroker()),
    )
    app.bot.running = False
    app.data_source_mode = mode
    app._live_operation_confirmed = live_confirmed
    app.mode_service.detect_bot_mode = MagicMock(return_value=mode)
    app.push_screen_wait = AsyncMock(return_value=warning_result)
    app.post_message = MagicMock()
    app._sync_state_from_bot = MagicMock()
    app.notify = MagicMock()
    return app


def _worker_service() -> MagicMock:
    service = MagicMock()
    service.run_bot_async.return_value = SimpleNamespace(
        state=WorkerState.RUNNING,
        name="test-worker",
        error=None,
    )
    return service


class TestBotLifecycleLiveStartGate:
    @pytest.mark.asyncio
    async def test_live_start_requires_confirmation_when_unconfirmed(self) -> None:
        app = _app_for_start_mode("live", live_confirmed=False, warning_result=True)
        worker_service = _worker_service()

        manager = BotLifecycleManager(app, worker_service=worker_service)

        await manager.start_bot()

        app.push_screen_wait.assert_awaited_once()
        assert app._live_operation_confirmed is True
        worker_service.run_bot_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_start_decline_prevents_bot_execution(self) -> None:
        app = _app_for_start_mode("live", live_confirmed=False, warning_result=False)
        worker_service = _worker_service()

        manager = BotLifecycleManager(app, worker_service=worker_service)

        await manager.start_bot()

        app.push_screen_wait.assert_awaited_once()
        assert app._live_operation_confirmed is False
        worker_service.run_bot_async.assert_not_called()
        app._sync_state_from_bot.assert_not_called()
        app.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_confirmed_live_start_does_not_prompt_again(self) -> None:
        app = _app_for_start_mode("live", live_confirmed=True)
        worker_service = _worker_service()

        manager = BotLifecycleManager(app, worker_service=worker_service)

        await manager.start_bot()

        app.push_screen_wait.assert_not_awaited()
        worker_service.run_bot_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_non_live_start_does_not_prompt(self) -> None:
        app = _app_for_start_mode("paper", live_confirmed=False)
        worker_service = _worker_service()

        manager = BotLifecycleManager(app, worker_service=worker_service)

        await manager.start_bot()

        app.push_screen_wait.assert_not_awaited()
        assert app._live_operation_confirmed is False
        worker_service.run_bot_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_paper_fallback_cross_checks_broker_live_mode(self) -> None:
        app = _app_for_start_mode("paper", live_confirmed=False, broker=RealBroker())
        worker_service = _worker_service()

        manager = BotLifecycleManager(app, worker_service=worker_service)

        await manager.start_bot()

        app.push_screen_wait.assert_awaited_once()
        assert app._live_operation_confirmed is True
        worker_service.run_bot_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_mode_service_error_preserves_broker_live_mode(self) -> None:
        app = _app_for_start_mode("paper", live_confirmed=False, broker=RealBroker())
        app.mode_service.detect_bot_mode.side_effect = RuntimeError("mode service unavailable")
        worker_service = _worker_service()

        manager = BotLifecycleManager(app, worker_service=worker_service)

        await manager.start_bot()

        app.push_screen_wait.assert_awaited_once()
        assert app._live_operation_confirmed is True
        worker_service.run_bot_async.assert_called_once()
