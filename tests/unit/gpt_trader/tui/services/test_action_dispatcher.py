"""Tests for ActionDispatcher service."""

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
        await dispatcher.toggle_bot()  # Should not raise


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


class TestShowAlertHistory:
    """Tests for show_alert_history action."""

    @pytest.mark.asyncio
    async def test_pushes_alert_history_screen(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """show_alert_history should push AlertHistoryScreen."""
        await dispatcher.show_alert_history()
        mock_app.push_screen.assert_called_once()
        # Verify the screen type
        screen_arg = mock_app.push_screen.call_args[0][0]
        from gpt_trader.tui.screens.alert_history import AlertHistoryScreen

        assert isinstance(screen_arg, AlertHistoryScreen)

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """show_alert_history should handle errors gracefully."""
        mock_app.push_screen.side_effect = Exception("Screen error")
        await dispatcher.show_alert_history()
        mock_app.notify.assert_called_once()
        assert "Error" in mock_app.notify.call_args[0][0]


class TestForceRefresh:
    """Tests for force_refresh action."""

    @pytest.mark.asyncio
    async def test_syncs_state_from_bot(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should sync state from bot."""
        mock_app.alert_manager = MagicMock()
        await dispatcher.force_refresh()
        mock_app.ui_coordinator.sync_state_from_bot.assert_called_once()

    @pytest.mark.asyncio
    async def test_updates_main_screen(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should update main screen."""
        mock_app.alert_manager = MagicMock()
        await dispatcher.force_refresh()
        mock_app.ui_coordinator.update_main_screen.assert_called_once()

    @pytest.mark.asyncio
    async def test_resets_alert_cooldowns(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should reset alert cooldowns."""
        mock_app.alert_manager = MagicMock()
        await dispatcher.force_refresh()
        mock_app.alert_manager.reset_cooldowns.assert_called_once()

    @pytest.mark.asyncio
    async def test_checks_alerts_after_refresh(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should check alerts after refresh."""
        mock_app.alert_manager = MagicMock()
        await dispatcher.force_refresh()
        mock_app.alert_manager.check_alerts.assert_called_once_with(mock_app.tui_state)

    @pytest.mark.asyncio
    async def test_notifies_user_of_completion(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should notify user on completion."""
        mock_app.alert_manager = MagicMock()
        await dispatcher.force_refresh()
        # Should be called at least twice (start and complete)
        assert mock_app.notify.call_count >= 2

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should handle errors gracefully."""
        mock_app.ui_coordinator.sync_state_from_bot.side_effect = Exception("Sync failed")
        await dispatcher.force_refresh()
        # Should notify about error
        assert any("failed" in str(call).lower() for call in mock_app.notify.call_args_list)

    @pytest.mark.asyncio
    async def test_works_without_alert_manager(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should work when alert_manager is not available."""
        # Remove alert_manager
        del mock_app.alert_manager
        # Should not raise
        await dispatcher.force_refresh()
        mock_app.ui_coordinator.sync_state_from_bot.assert_called_once()


class TestEnableReduceOnly:
    """Tests for enable_reduce_only action."""

    @pytest.mark.asyncio
    async def test_enable_reduce_only_calls_risk_manager(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """enable_reduce_only should call risk_manager.set_reduce_only_mode."""
        mock_app.bot.risk_manager = MagicMock()
        mock_app.bot.risk_manager.reduce_only_mode = False
        mock_app.bot.risk_manager.set_reduce_only_mode = MagicMock()

        await dispatcher.enable_reduce_only()

        mock_app.bot.risk_manager.set_reduce_only_mode.assert_called_once_with(
            True, reason="operator_reduce_only"
        )

    @pytest.mark.asyncio
    async def test_enable_reduce_only_notifies_already_enabled(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """enable_reduce_only should notify if already in reduce-only mode."""
        mock_app.bot.risk_manager = MagicMock()
        mock_app.bot.risk_manager.reduce_only_mode = True

        await dispatcher.enable_reduce_only()

        mock_app.notify.assert_called_once()
        assert "Already" in mock_app.notify.call_args[0][0]

    @pytest.mark.asyncio
    async def test_enable_reduce_only_no_risk_manager(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """enable_reduce_only should warn when no risk manager available."""
        mock_app.bot = None

        await dispatcher.enable_reduce_only()

        mock_app.notify.assert_called_once()
        assert "No risk manager" in mock_app.notify.call_args[0][0]
