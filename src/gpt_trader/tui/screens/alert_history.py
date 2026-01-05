"""Alert History Screen for TUI.

Displays a history of triggered alerts with timestamps and severities.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import Screen
from textual.widgets import Button, DataTable, Label, Static

from gpt_trader.tui.notification_helpers import notify_success
from gpt_trader.tui.widgets import ContextualFooter
from gpt_trader.tui.widgets.alert_inbox import get_recovery_hint
from gpt_trader.tui.widgets.shell import CommandBar
from gpt_trader.tui.widgets.table_copy_mixin import TableCopyMixin
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.services.alert_manager import Alert, AlertCategory

logger = get_logger(__name__, component="tui")


class AlertHistoryScreen(TableCopyMixin, Screen):
    """Screen displaying alert history.

    Shows all triggered alerts with timestamps, severities, and messages.
    Allows clearing the history and acknowledging alerts.

    Keyboard shortcuts:
        x: Clear history
        y: Copy selected row
        Y: Copy all rows
        r: Reset cooldowns
        1-5: Filter by category
    """

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("x", "clear_history", "Clear"),
        ("y", "copy_row", "Copy"),
        ("Y", "copy_all", "Copy All"),
        ("r", "reset_cooldowns", "Reset"),
        ("1", "filter_all", "All"),
        ("2", "filter_trade", "Trade"),
        ("3", "filter_system", "System"),
        ("4", "filter_risk", "Risk"),
        ("5", "filter_error", "Error"),
    ]

    # Current category filter (None = all categories)
    _current_filter: str = "all"

    # Styles moved to styles/screens/alert_history.tcss

    def compose(self) -> ComposeResult:
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )
        with Container():
            yield Label("Alert History", classes="title")

            # Filter chips row
            with Horizontal(classes="filter-chips"):
                yield Button("All [1]", id="filter-all", classes="filter-chip active")
                yield Button("Trade [2]", id="filter-trade", classes="filter-chip")
                yield Button("System [3]", id="filter-system", classes="filter-chip")
                yield Button("Risk [4]", id="filter-risk", classes="filter-chip")
                yield Button("Error [5]", id="filter-error", classes="filter-chip")

            with Static(classes="stats"):
                yield Label("", id="stats-label")

            yield DataTable(id="alert-table")

            with Static(classes="button-bar"):
                yield Button("Clear [X]", id="clear-btn", variant="warning")
                yield Button("Reset [R]", id="reset-btn", variant="default")
                yield Button("Close [ESC]", id="close-btn", variant="primary")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize the alert table on mount."""
        table = self.query_one("#alert-table", DataTable)

        # Add columns
        table.add_column("Time", width=18)
        table.add_column("Sev", width=8)
        table.add_column("Cat", width=6)
        table.add_column("Title", width=18)
        table.add_column("Message", width=40)
        table.add_column("Action", width=16)

        # Populate with alert history
        self._refresh_table()

    def _refresh_table(self) -> None:
        """Refresh the alert table from history."""
        from gpt_trader.tui.services.alert_manager import AlertCategory
        from gpt_trader.tui.theme import THEME

        table = self.query_one("#alert-table", DataTable)
        stats_label = self.query_one("#stats-label", Label)

        # Clear existing rows
        table.clear()

        # Get alert history from AlertManager
        all_alerts: list[Alert] = []
        if hasattr(self.app, "alert_manager"):
            all_alerts = self.app.alert_manager.get_history(limit=100)

        # Calculate category counts for stats
        category_counts: dict[str, int] = {}
        for alert in all_alerts:
            cat = alert.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Filter alerts by current filter
        alerts = self._filter_alerts(all_alerts)

        # Update stats with category counts
        if all_alerts:
            stats_parts = [f"Total: {len(all_alerts)}"]
            # Show category counts in order
            for cat in ["trade", "position", "strategy", "risk", "system", "error"]:
                if cat in category_counts:
                    stats_parts.append(f"{cat.upper()[:3]}: {category_counts[cat]}")
            stats_label.update(" | ".join(stats_parts))
        else:
            stats_label.update("No alerts triggered")

        # Add alerts to table (most recent first)
        for alert in alerts:
            # Format timestamp (alert.timestamp is a float from time.time())
            dt = datetime.fromtimestamp(alert.timestamp)
            time_str = dt.strftime("%m-%d %H:%M:%S")

            # Color-code severity
            severity_text = Text(alert.severity.value.upper()[:4])
            if alert.severity.value == "error":
                severity_text.stylize(f"{THEME.colors.error} bold")
            elif alert.severity.value == "warning":
                severity_text.stylize(THEME.colors.warning)
            else:
                severity_text.stylize(THEME.colors.info)

            # Color-code category
            category_text = Text(alert.category.value.upper()[:3])
            category_colors = {
                "trade": THEME.colors.success,
                "position": THEME.colors.primary,
                "strategy": THEME.colors.accent,
                "risk": THEME.colors.warning,
                "system": THEME.colors.muted,
                "error": THEME.colors.error,
            }
            cat_color = category_colors.get(alert.category.value, THEME.colors.muted)
            category_text.stylize(cat_color)

            # Get recovery hint
            hint = get_recovery_hint(alert.rule_id, alert.category.value) or ""
            hint_text = Text(hint)
            if hint.startswith("["):
                hint_text.stylize(f"{THEME.colors.accent} bold")

            # Truncate message
            msg = alert.message[:38] + ".." if len(alert.message) > 40 else alert.message

            table.add_row(
                time_str,
                severity_text,
                category_text,
                alert.title[:16] + ".." if len(alert.title) > 18 else alert.title,
                msg,
                hint_text,
                key=f"alert-{alert.timestamp}",
            )

    def _filter_alerts(self, alerts: list[Alert]) -> list[Alert]:
        """Filter alerts by current filter setting.

        Args:
            alerts: List of alerts to filter.

        Returns:
            Filtered list of alerts.
        """
        if self._current_filter == "all":
            return alerts

        from gpt_trader.tui.services.alert_manager import AlertCategory

        filter_categories = {
            "trade": {AlertCategory.TRADE, AlertCategory.POSITION},
            "system": {AlertCategory.SYSTEM, AlertCategory.STRATEGY},
            "risk": {AlertCategory.RISK},
            "error": {AlertCategory.ERROR},
        }

        allowed = filter_categories.get(self._current_filter, set())
        return [a for a in alerts if a.category in allowed]

    def action_clear_history(self) -> None:
        """Clear all alert history."""
        if hasattr(self.app, "alert_manager"):
            self.app.alert_manager.clear_history()
            self._refresh_table()
            notify_success(self.app, "Alert history cleared")

    def action_reset_cooldowns(self) -> None:
        """Reset all alert cooldowns."""
        if hasattr(self.app, "alert_manager"):
            self.app.alert_manager.reset_cooldowns()
            notify_success(self.app, "Alert cooldowns reset — alerts can trigger again")

    def action_filter_all(self) -> None:
        """Show all alerts."""
        self._set_filter("all")

    def action_filter_trade(self) -> None:
        """Show trade/position alerts."""
        self._set_filter("trade")

    def action_filter_system(self) -> None:
        """Show system/strategy alerts."""
        self._set_filter("system")

    def action_filter_risk(self) -> None:
        """Show risk alerts."""
        self._set_filter("risk")

    def action_filter_error(self) -> None:
        """Show error alerts."""
        self._set_filter("error")

    def _set_filter(self, filter_name: str) -> None:
        """Set the active filter and refresh.

        Args:
            filter_name: Filter to activate (all, trade, system, risk, error).
        """
        self._current_filter = filter_name
        self._update_filter_chips()
        self._refresh_table()

    def _update_filter_chips(self) -> None:
        """Update filter chip styles to show active state."""
        filter_ids = ["filter-all", "filter-trade", "filter-system", "filter-risk", "filter-error"]
        for chip_id in filter_ids:
            try:
                chip = self.query_one(f"#{chip_id}", Button)
                expected_id = f"filter-{self._current_filter}"
                if chip_id == expected_id:
                    chip.add_class("active")
                else:
                    chip.remove_class("active")
            except Exception:
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "clear-btn":
            self.action_clear_history()
        elif button_id == "reset-btn":
            self.action_reset_cooldowns()
        elif button_id == "close-btn":
            self.app.pop_screen()
        elif button_id == "filter-all":
            self.action_filter_all()
        elif button_id == "filter-trade":
            self.action_filter_trade()
        elif button_id == "filter-system":
            self.action_filter_system()
        elif button_id == "filter-risk":
            self.action_filter_risk()
        elif button_id == "filter-error":
            self.action_filter_error()
