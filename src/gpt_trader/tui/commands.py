"""
Command palette commands for GPT-Trader TUI.

Provides searchable commands for the Textual command palette,
allowing users to quickly access actions via Ctrl+P.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.command import Hit, Hits, Provider

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp


class TraderCommands(Provider):
    """Command provider for GPT-Trader TUI actions.

    Exposes all major TUI actions through the command palette,
    organized by category for easy discovery.
    """

    @property
    def app(self) -> TraderApp:
        """Get the app instance with proper typing."""
        return self.screen.app  # type: ignore

    async def search(self, query: str) -> Hits:
        """Search for commands matching the query.

        Args:
            query: The search query string.

        Yields:
            Hit objects for matching commands.
        """
        matcher = self.matcher(query)

        # Bot control commands
        commands = [
            # Bot Operations
            ("Start/Stop Bot", "Toggle the trading bot on/off", "toggle_bot", "s"),
            ("Reconnect Data", "Reconnect to data source", "reconnect_data", "r"),
            ("PANIC Stop", "Emergency stop all trading", "panic", "p"),
            ("Force Refresh", "Force refresh bot state and UI", "force_refresh", "f"),
            # Navigation
            ("Show Market", "Open market overview screen", "show_market", "m"),
            ("Show Details", "Open detailed view screen", "show_details", "d"),
            ("Focus Logs", "Focus the log panel", "focus_logs", "l"),
            ("Show Full Logs", "Open full logs screen", "show_full_logs", "1"),
            ("Show System Details", "Open system details screen", "show_system_details", "2"),
            ("Show Help", "Open help screen", "show_help", "?"),
            # Configuration
            ("Show Config", "Open configuration modal", "show_config", "c"),
            ("Show Mode Info", "Show current operating mode", "show_mode_info", "i"),
            ("Show Alerts", "Open alert history", "show_alerts", "a"),
            # Watchlist
            ("Edit Watchlist", "Add/remove symbols from watchlist", "edit_watchlist", "w"),
            ("Clear Watchlist", "Remove all symbols from watchlist", "clear_watchlist", ""),
            # Operator/Diagnostics
            ("Diagnose Connection", "Run connection diagnostics", "diagnose_connection", ""),
            ("Export Logs", "Export logs to file", "export_logs", ""),
            ("Export State", "Export current state snapshot", "export_state", ""),
            ("Reconnect WebSocket", "Force WebSocket reconnection", "reconnect_websocket", ""),
            ("Reset Circuit Breakers", "Reset all circuit breakers", "reset_circuit_breakers", ""),
            # Theme
            ("Toggle Theme", "Switch between dark and light theme", "toggle_theme", "t"),
            # Log Levels
            ("Log Level: DEBUG", "Set log level to DEBUG", "set_log_level_debug", "Ctrl+1"),
            ("Log Level: INFO", "Set log level to INFO", "set_log_level_info", "Ctrl+2"),
            ("Log Level: WARNING", "Set log level to WARNING", "set_log_level_warning", "Ctrl+3"),
            ("Log Level: ERROR", "Set log level to ERROR", "set_log_level_error", "Ctrl+4"),
            # Application
            ("Quit", "Exit the application", "quit", "q"),
        ]

        for name, description, action, shortcut in commands:
            # Include shortcut in searchable text
            search_text = f"{name} {description} {shortcut}"
            score = matcher.match(search_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(name),
                    partial=self._create_action_callback(action),
                    help=f"{description} [{shortcut}]",
                )

    def _create_action_callback(self, action: str) -> callable:
        """Create a callback for the given action.

        Args:
            action: The action name.

        Returns:
            A callable that executes the action.
        """

        async def callback() -> None:
            # Special handling for log level actions
            if action == "set_log_level_debug":
                self.app.action_set_log_level("DEBUG")
            elif action == "set_log_level_info":
                self.app.action_set_log_level("INFO")
            elif action == "set_log_level_warning":
                self.app.action_set_log_level("WARNING")
            elif action == "set_log_level_error":
                self.app.action_set_log_level("ERROR")
            else:
                # Use run_action for standard actions
                await self.app.run_action(action)

        return callback
