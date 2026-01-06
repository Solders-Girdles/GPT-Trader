"""Actions mixin for TraderApp.

Contains action methods and event handlers for user interactions.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from gpt_trader.tui.notification_helpers import notify_action
from gpt_trader.tui.screens import DetailsScreen, MarketScreen, StrategyDetailScreen
from gpt_trader.tui.services.execution_telemetry import get_execution_telemetry
from gpt_trader.tui.widgets import SlimStatusWidget
from gpt_trader.tui.widgets.execution_issues_modal import ExecutionIssuesModal
from gpt_trader.tui.widgets.status import BotStatusWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class TraderAppActionsMixin:
    """Mixin providing action methods for TraderApp.

    Action methods:
    - action_toggle_bot: Toggle bot running state
    - action_show_config: Show configuration modal
    - action_focus_logs: Focus the log widget
    - action_set_log_level: Set log level
    - action_show_full_logs: Show full logs screen
    - action_show_system_details: Show system details screen
    - action_show_mode_info: Show mode information modal
    - action_show_market: Show market overlay
    - action_show_details: Show details overlay
    - action_show_strategy: Show strategy detail screen
    - action_show_help: Show help screen
    - action_reconnect_data: Reconnect data source
    - action_panic: Show panic confirmation modal
    - action_toggle_theme: Toggle theme
    - action_show_alerts: Show alert history
    - action_force_refresh: Force refresh UI
    - action_toggle_performance: Toggle performance overlay
    - action_quit: Quit application

    Event handlers:
    - on_bot_status_widget_toggle_bot_pressed
    - on_slim_status_widget_toggle_bot_pressed
    - on_slim_status_widget_mode_changed
    - on_mode_selector_mode_changed
    """

    # Type hints for attributes from TraderApp
    if TYPE_CHECKING:
        action_dispatcher: Any

        def push_screen(self, screen: Any) -> None: ...
        def pop_screen(self) -> None: ...
        def query_one(self, selector: str, widget_type: type | None = None) -> Any: ...
        async def action_toggle_bot(self) -> None: ...
        async def _switch_to_mode(self, mode: str) -> bool: ...

    async def action_toggle_bot(self: TraderApp) -> None:
        """Toggle bot running state."""
        await self.action_dispatcher.toggle_bot()

    async def action_show_config(self: TraderApp) -> None:
        """Show configuration modal."""
        await self.action_dispatcher.show_config()

    async def action_focus_logs(self: TraderApp) -> None:
        """Focus the log widget."""
        await self.action_dispatcher.focus_logs()

    def action_set_log_level(self: TraderApp, level: str) -> None:
        """Set log level via keyboard shortcut.

        Args:
            level: Log level name (DEBUG, INFO, WARNING, ERROR)
        """
        from gpt_trader.tui.widgets.logs import LogWidget

        level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
        try:
            log_widget = self.query_one("#dash-logs", LogWidget)
            log_widget.set_level(level_map.get(level, 20))
            notify_action(self, f"Log level: {level}")
        except Exception:
            pass

    async def action_show_full_logs(self: TraderApp) -> None:
        """Show full logs screen."""
        await self.action_dispatcher.show_full_logs()

    async def action_show_system_details(self: TraderApp) -> None:
        """Show system details screen."""
        await self.action_dispatcher.show_system_details()

    async def action_show_exec_issues(self: TraderApp) -> None:
        """Show execution issues modal."""
        metrics = get_execution_telemetry().get_metrics()
        self.push_screen(ExecutionIssuesModal(metrics))

    async def action_show_mode_info(self: TraderApp) -> None:
        """Show mode information modal."""
        await self.action_dispatcher.show_mode_info()

    async def action_show_market(self: TraderApp) -> None:
        """Show market overlay screen."""
        self.push_screen(MarketScreen())

    async def action_show_details(self: TraderApp) -> None:
        """Show details overlay screen."""
        self.push_screen(DetailsScreen())

    async def action_show_strategy(self: TraderApp) -> None:
        """Show strategy detail screen."""
        self.push_screen(StrategyDetailScreen())

    async def action_show_help(self: TraderApp) -> None:
        """Show help screen."""
        await self.action_dispatcher.show_help()

    async def action_reconnect_data(self: TraderApp) -> None:
        """Reconnect data source."""
        await self.action_dispatcher.reconnect_data()

    async def action_panic(self: TraderApp) -> None:
        """Show panic confirmation modal."""
        await self.action_dispatcher.panic()

    async def action_toggle_theme(self: TraderApp) -> None:
        """Toggle theme."""
        await self.action_dispatcher.toggle_theme()

    async def action_show_alerts(self: TraderApp) -> None:
        """Show alert history screen."""
        await self.action_dispatcher.show_alert_history()

    async def action_force_refresh(self: TraderApp) -> None:
        """Force refresh bot state and UI."""
        await self.action_dispatcher.force_refresh()

    async def action_toggle_performance(self: TraderApp) -> None:
        """Toggle performance monitoring overlay.

        Shows real-time TUI performance metrics including FPS, latency,
        memory usage, and throttler efficiency.
        """
        from gpt_trader.tui.screens.performance_screen import PerformanceScreen

        try:
            # Check if already showing performance screen
            self.query_one(PerformanceScreen)
            self.pop_screen()
        except Exception:
            # Not showing - push performance screen
            self.push_screen(PerformanceScreen())

    async def action_quit(self: TraderApp) -> None:
        """Quit the application."""
        await self.action_dispatcher.quit_app()

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def on_bot_status_widget_toggle_bot_pressed(
        self: TraderApp, message: BotStatusWidget.ToggleBotPressed
    ) -> None:
        """Handle start/stop button press from BotStatusWidget."""
        asyncio.create_task(self.action_toggle_bot())

    def on_slim_status_widget_toggle_bot_pressed(
        self: TraderApp, message: SlimStatusWidget.ToggleBotPressed
    ) -> None:
        """Handle start/stop button press from SlimStatusWidget."""
        asyncio.create_task(self.action_toggle_bot())

    def on_slim_status_widget_mode_changed(
        self: TraderApp, message: SlimStatusWidget.ModeChanged
    ) -> None:
        """Handle mode change request from SlimStatusWidget."""
        asyncio.create_task(self._switch_to_mode(message.mode))

    def on_mode_selector_mode_changed(self: TraderApp, message: Any) -> None:
        """Handle mode change request from ModeSelector."""
        from gpt_trader.tui.widgets import ModeSelector

        if isinstance(message, ModeSelector.ModeChanged):
            asyncio.create_task(self._switch_to_mode(message.new_mode))
