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


class TestShowAlertHistory:
    """Tests for show_alert_history action."""

    @pytest.mark.asyncio
    async def test_pushes_alert_history_screen(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """show_alert_history should push AlertHistoryScreen."""
        await dispatcher.show_alert_history()
        mock_app.push_screen.assert_called_once()
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
        assert mock_app.notify.call_count >= 2

    @pytest.mark.asyncio
    async def test_handles_error_gracefully(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should handle errors gracefully."""
        mock_app.ui_coordinator.sync_state_from_bot.side_effect = Exception("Sync failed")
        await dispatcher.force_refresh()
        assert any("failed" in str(call).lower() for call in mock_app.notify.call_args_list)

    @pytest.mark.asyncio
    async def test_works_without_alert_manager(
        self, dispatcher: ActionDispatcher, mock_app: MagicMock
    ) -> None:
        """force_refresh should work when alert_manager is not available."""
        del mock_app.alert_manager
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
