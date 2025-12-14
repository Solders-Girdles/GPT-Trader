"""Alert History Screen for TUI.

Displays a history of triggered alerts with timestamps and severities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Static

from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.tui.widgets import ContextualFooter
from gpt_trader.tui.widgets.shell import CommandBar

if TYPE_CHECKING:
    from gpt_trader.tui.services.alert_manager import Alert

logger = get_logger(__name__, component="tui")


class AlertHistoryScreen(Screen):
    """Screen displaying alert history.

    Shows all triggered alerts with timestamps, severities, and messages.
    Allows clearing the history and acknowledging alerts.
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("c", "clear_history", "Clear History"),
        ("r", "reset_cooldowns", "Reset Cooldowns"),
    ]

    # Styles moved to styles/screens/alert_history.tcss

    def compose(self) -> ComposeResult:
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )
        with Container():
            yield Label("Alert History", classes="title")

            with Static(classes="stats"):
                yield Label("", id="stats-label")

            yield DataTable(id="alert-table")

            with Static(classes="button-bar"):
                yield Button("Clear History [C]", id="clear-btn", variant="warning")
                yield Button("Reset Cooldowns [R]", id="reset-btn", variant="default")
                yield Button("Close [ESC]", id="close-btn", variant="primary")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize the alert table on mount."""
        table = self.query_one("#alert-table", DataTable)

        # Add columns
        table.add_column("Time", width=20)
        table.add_column("Severity", width=12)
        table.add_column("Rule", width=20)
        table.add_column("Message", width=60)

        # Populate with alert history
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the alert table from history."""
        from gpt_trader.tui.theme import THEME

        table = self.query_one("#alert-table", DataTable)
        stats_label = self.query_one("#stats-label", Label)

        # Clear existing rows
        table.clear()

        # Get alert history from AlertManager
        alerts: list[Alert] = []
        if hasattr(self.app, "alert_manager"):
            alerts = self.app.alert_manager.get_history()

        # Update stats
        if alerts:
            severity_counts = {}
            for alert in alerts:
                severity_counts[alert.severity.value] = (
                    severity_counts.get(alert.severity.value, 0) + 1
                )
            stats_text = f"Total: {len(alerts)} alerts | "
            stats_text += " | ".join(
                f"{sev.upper()}: {count}" for sev, count in severity_counts.items()
            )
            stats_label.update(stats_text)
        else:
            stats_label.update("No alerts triggered")

        # Add alerts to table (most recent first)
        for alert in reversed(alerts):
            # Format timestamp
            time_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")

            # Color-code severity
            severity_text = Text(alert.severity.value.upper())
            if alert.severity.value == "error":
                severity_text.stylize(f"{THEME.colors.error} bold")
            elif alert.severity.value == "warning":
                severity_text.stylize(THEME.colors.warning)
            else:
                severity_text.stylize(THEME.colors.info)

            table.add_row(
                time_str,
                severity_text,
                alert.rule_id,
                alert.message[:60] + "..." if len(alert.message) > 60 else alert.message,
                key=f"alert-{alert.timestamp.timestamp()}",
            )

    def action_clear_history(self) -> None:
        """Clear all alert history."""
        if hasattr(self.app, "alert_manager"):
            self.app.alert_manager.clear_history()
            self._refresh_table()
            self.app.notify("Alert history cleared", severity="information")

    def action_reset_cooldowns(self) -> None:
        """Reset all alert cooldowns."""
        if hasattr(self.app, "alert_manager"):
            self.app.alert_manager.reset_cooldowns()
            self.app.notify(
                "Alert cooldowns reset - alerts can trigger again",
                severity="information",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "clear-btn":
            self.action_clear_history()
        elif button_id == "reset-btn":
            self.action_reset_cooldowns()
        elif button_id == "close-btn":
            self.app.pop_screen()
