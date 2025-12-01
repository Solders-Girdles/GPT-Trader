"""
Main TUI Application for GPT-Trader.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App

from gpt_trader.tui.screens import MainScreen
from gpt_trader.tui.widgets import ConfigModal
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.orchestration.trading_bot.bot import TradingBot

logger = get_logger(__name__, component="tui")


class TraderApp(App):
    """GPT-Trader Terminal User Interface."""

    CSS_PATH = "styles.tcss"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "toggle_bot", "Start/Stop Bot"),
        ("c", "show_config", "Config"),
        ("l", "focus_logs", "Focus Logs"),
    ]

    def __init__(self, bot: TradingBot) -> None:
        super().__init__()
        self.bot = bot
        self._update_task: asyncio.Task | None = None

        # Initialize State
        from gpt_trader.tui.state import TuiState

        self.tui_state = TuiState()

    async def on_mount(self) -> None:
        """Called when app starts."""
        self.push_screen(MainScreen())
        self.log("TUI Started")

        # Connect to StatusReporter observer if available
        if hasattr(self.bot.engine, "status_reporter"):
            self.bot.engine.status_reporter.add_observer(self._on_status_update)

        # Start the bot in background if not already running
        if not self.bot.running:
            asyncio.create_task(self.bot.run())

        # Start UI update loop (fallback/backup and for fields not in status reporter)
        self._update_task = asyncio.create_task(self._update_loop())

        # Bind state to widgets
        self._bind_state()

    async def on_unmount(self) -> None:
        """Called when app stops."""
        if hasattr(self.bot.engine, "status_reporter"):
            self.bot.engine.status_reporter.remove_observer(self._on_status_update)

    def _on_status_update(self, status: dict) -> None:
        """Callback for StatusReporter updates."""
        # This might be called from a background thread or loop
        # Schedule the update on the main thread
        self.call_from_thread(self._apply_status_update, status)

    def _apply_status_update(self, status: dict) -> None:
        """Apply status update to state and UI."""
        # Update State
        runtime_state = None
        if hasattr(self.bot.engine.context, "runtime_state"):
            runtime_state = self.bot.engine.context.runtime_state

        self.tui_state.running = self.bot.running
        self.tui_state.update_from_bot_status(status, runtime_state)

        # Update UI
        try:
            main_screen = self.query_one(MainScreen)
            main_screen.update_ui(self.tui_state)
        except Exception:
            pass

    def _bind_state(self) -> None:
        """Bind reactive state to widgets."""
        # This is where we could set up direct bindings if widgets supported it
        # For now, we'll just rely on the update loop pushing data to state,
        # and then we can push state to widgets or have widgets watch state.
        # To keep it simple for this refactor, we will manually update widgets from state in _update_ui
        # but the source of truth is now self.tui_state
        pass

    async def _update_loop(self) -> None:
        """Periodically update UI from bot state (Fallback loop)."""
        while True:
            try:
                # If we don't have observers, we must poll
                if (
                    not hasattr(self.bot.engine, "status_reporter")
                    or not self.bot.engine.status_reporter._observers
                ):
                    self._sync_state_from_bot()
                    try:
                        main_screen = self.query_one(MainScreen)
                        main_screen.update_ui(self.tui_state)
                    except Exception:
                        pass
            except Exception as e:
                self.log(f"UI Update Error: {e}")
            await asyncio.sleep(1)

    def _sync_state_from_bot(self) -> None:
        """Fetch state from bot and update TuiState."""
        self.tui_state.running = self.bot.running

        # Access runtime state safely
        runtime_state = None
        if hasattr(self.bot.engine.context, "runtime_state"):
            runtime_state = self.bot.engine.context.runtime_state

        # Access StatusReporter for data
        status = {}
        if hasattr(self.bot.engine, "status_reporter"):
            status = self.bot.engine.status_reporter.get_status()

        self.tui_state.update_from_bot_status(status, runtime_state)

    async def action_toggle_bot(self) -> None:
        """Toggle bot running state."""
        if self.bot.running:
            await self.bot.stop()
        else:
            asyncio.create_task(self.bot.run())

    async def action_show_config(self) -> None:
        """Show configuration modal."""
        self.push_screen(ConfigModal(self.bot.config))

    async def action_focus_logs(self) -> None:
        """Focus the log widget."""
        # Focus the full log widget and switch tab if needed
        try:
            self.query_one("#logs-full").focus()
        except Exception:
            pass

    async def action_quit(self) -> None:
        """Quit the application."""
        if self.bot.running:
            await self.bot.stop()
        self.exit()
