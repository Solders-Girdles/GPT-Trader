from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gpt_trader.tui.services.action_dispatcher import ActionDispatcher


@pytest.fixture
def mock_app() -> MagicMock:
    """Create a mock TraderApp instance."""
    app = MagicMock()
    app.lifecycle_manager = MagicMock()
    app.lifecycle_manager.toggle_bot = AsyncMock()
    app.config_service = MagicMock()
    app.ui_coordinator = MagicMock()
    app.theme_service = MagicMock()
    app.bot = MagicMock()
    app.bot.running = False
    app.data_source_mode = "demo"
    app.tui_state = MagicMock()
    app.tui_state.check_connection_health = MagicMock(return_value=True)
    app.notify = MagicMock()
    app.push_screen = MagicMock()
    app.query_one = MagicMock()
    app.exit = MagicMock()
    app.screen = MagicMock()
    return app


@pytest.fixture
def dispatcher(mock_app: MagicMock) -> ActionDispatcher:
    """Create an ActionDispatcher instance with mock app."""
    return ActionDispatcher(mock_app)


class TestToggleBot:
    """Tests for toggle_bot action."""

    @pytest.mark.asyncio
    async def test_delegates_to_lifecycle_manager(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """toggle_bot should delegate to lifecycle_manager."""
        await dispatcher.toggle_bot()
        mock_app.lifecycle_manager.toggle_bot.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_op_when_no_lifecycle_manager(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """toggle_bot should be a no-op when lifecycle_manager is None."""
        mock_app.lifecycle_manager = None
        await dispatcher.toggle_bot()
        assert mock_app.lifecycle_manager is None
        mock_app.notify.assert_not_called()


class TestShowConfig:
    """Tests for show_config action."""

    @pytest.mark.asyncio
    async def test_shows_config_modal_when_bot_has_config(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """show_config should show modal when bot has config."""
        mock_app.bot.config = {"key": "value"}
        await dispatcher.show_config()
        mock_app.config_service.show_config_modal.assert_called_once_with(mock_app.bot.config)

    @pytest.mark.asyncio
    async def test_notifies_when_no_config(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """show_config should notify when no config available."""
        mock_app.bot = None
        await dispatcher.show_config()
        mock_app.notify.assert_called_once()
        assert "No configuration" in mock_app.notify.call_args[0][0]


class TestToggleTheme:
    """Tests for toggle_theme action."""

    @pytest.mark.asyncio
    async def test_delegates_to_theme_service(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """toggle_theme should delegate to theme_service."""
        await dispatcher.toggle_theme()
        mock_app.theme_service.toggle_theme.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_theme_service_error(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """toggle_theme should handle errors gracefully."""
        mock_app.theme_service.toggle_theme.side_effect = Exception("Theme error")
        await dispatcher.toggle_theme()
        mock_app.notify.assert_called_once()
        assert "failed" in mock_app.notify.call_args[0][0].lower()


class TestReconnectData:
    """Tests for reconnect_data action."""

    @pytest.mark.asyncio
    async def test_no_op_in_demo_mode(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """reconnect_data should be a no-op in demo mode."""
        mock_app.data_source_mode = "demo"
        await dispatcher.reconnect_data()
        mock_app.notify.assert_called_once()
        assert "simulated data" in mock_app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_syncs_state_in_paper_mode(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """reconnect_data should sync state in paper mode."""
        mock_app.data_source_mode = "paper"
        await dispatcher.reconnect_data()
        mock_app.ui_coordinator.sync_state_from_bot.assert_called_once()


class TestQuitApp:
    """Tests for quit_app action."""

    @pytest.mark.asyncio
    async def test_stops_running_bot(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """quit_app should stop running bot."""
        mock_app.bot.running = True
        mock_app.lifecycle_manager.stop_bot = AsyncMock()
        await dispatcher.quit_app()
        mock_app.lifecycle_manager.stop_bot.assert_called_once()
        mock_app.exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_exits_when_bot_stopped(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """quit_app should exit when bot already stopped."""
        mock_app.bot.running = False
        await dispatcher.quit_app()
        mock_app.exit.assert_called_once()

    @pytest.mark.asyncio
    async def test_exits_even_on_error(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """quit_app should exit even if cleanup fails."""
        mock_app.bot.running = True
        mock_app.lifecycle_manager.stop_bot = AsyncMock(side_effect=Exception("Stop failed"))
        await dispatcher.quit_app()
        mock_app.exit.assert_called_once()
