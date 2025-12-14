"""
Alert Inbox Widget for displaying recent notifications.

Provides a compact, filterable view of alerts categorized by type,
aligned with system log levels for consistent filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Label, ListView, ListItem, Static

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.services.alert_manager import (
        Alert,
        AlertCategory,
        AlertManager,
        AlertSeverity,
    )

logger = get_logger(__name__, component="tui")


class AlertItem(ListItem):
    """Single alert item in the inbox list."""

    def __init__(self, alert: Alert, **kwargs) -> None:
        """Initialize alert item.

        Args:
            alert: The alert to display.
        """
        super().__init__(**kwargs)
        self.alert = alert

    def compose(self) -> ComposeResult:
        """Compose the alert item."""
        # Severity indicator
        severity_class = f"severity-{self.alert.severity.value}"
        category_class = f"category-{self.alert.category.value}"

        with Horizontal(classes=f"alert-item {severity_class} {category_class}"):
            # Severity indicator dot
            yield Label("●", classes=f"severity-dot {severity_class}")

            # Alert content
            with Vertical(classes="alert-content"):
                yield Label(self.alert.title, classes="alert-title")
                yield Label(
                    self._truncate(self.alert.message, 50),
                    classes="alert-message",
                )

            # Category badge
            yield Label(
                self.alert.category.value.upper()[:3],
                classes=f"category-badge {category_class}",
            )

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."


class AlertInbox(Static):
    """Compact alert inbox showing recent notifications.

    Features:
    - Shows recent alerts from AlertManager
    - Filterable by category (aligned with log levels)
    - Filterable by minimum severity
    - Quick filter buttons for common views

    Keyboard:
    - 1: Show all alerts
    - 2: Show trade alerts only
    - 3: Show system alerts only
    - 4: Show errors only
    - c: Clear alert history
    """

    BINDINGS = [
        Binding("1", "filter_all", "All", show=True),
        Binding("2", "filter_trade", "Trades", show=True),
        Binding("3", "filter_system", "System", show=True),
        Binding("4", "filter_error", "Errors", show=True),
        Binding("c", "clear_alerts", "Clear", show=False),
    ]

    DEFAULT_CSS = """
    AlertInbox {
        height: auto;
        max-height: 20;
        border: solid $border-subtle;
        background: $bg-primary;
    }

    .inbox-header {
        height: 1;
        width: 100%;
        background: $bg-elevated;
        padding: 0 1;
    }

    .inbox-header .widget-header {
        width: 1fr;
        text-style: bold;
        color: $accent;
    }

    .alert-count {
        color: $text-muted;
    }

    .inbox-filters {
        height: 1;
        width: 100%;
        padding: 0 1;
        background: $bg-secondary;
    }

    .filter-btn {
        min-width: 8;
        height: 1;
        margin-right: 1;
        background: transparent;
        color: $text-muted;
        border: none;
    }

    .filter-btn.active {
        color: $accent;
        text-style: bold;
    }

    .filter-btn:hover {
        background: $bg-elevated;
    }

    #alert-list {
        height: auto;
        max-height: 15;
        padding: 0;
    }

    .alert-item {
        height: 2;
        width: 100%;
        padding: 0 1;
    }

    .alert-item:hover {
        background: $bg-elevated;
    }

    .severity-dot {
        width: 2;
        text-align: center;
    }

    .severity-dot.severity-information {
        color: $primary;
    }

    .severity-dot.severity-warning {
        color: $warning;
    }

    .severity-dot.severity-error {
        color: $error;
    }

    .alert-content {
        width: 1fr;
        height: auto;
    }

    .alert-title {
        text-style: bold;
        color: $text;
    }

    .alert-message {
        color: $text-muted;
    }

    .category-badge {
        width: 5;
        text-align: center;
        padding: 0 1;
        color: $text-muted;
    }

    .category-badge.category-trade {
        color: $success;
    }

    .category-badge.category-position {
        color: $primary;
    }

    .category-badge.category-strategy {
        color: $accent;
    }

    .category-badge.category-risk {
        color: $warning;
    }

    .category-badge.category-system {
        color: $text-muted;
    }

    .category-badge.category-error {
        color: $error;
    }

    .empty-inbox {
        height: 3;
        width: 100%;
        align: center middle;
        color: $text-muted;
    }
    """

    # Current filter state - using Any to avoid import issues
    # At runtime these will be set[AlertCategory] and AlertSeverity | None
    category_filter: reactive[set[Any]] = reactive(set, init=False)
    min_severity: reactive[Any] = reactive(None, init=False)
    active_filter: reactive[str] = reactive("all")

    def __init__(
        self,
        alert_manager: AlertManager | None = None,
        max_alerts: int = 10,
        **kwargs,
    ) -> None:
        """Initialize AlertInbox.

        Args:
            alert_manager: AlertManager instance to pull alerts from.
            max_alerts: Maximum alerts to display.
        """
        super().__init__(**kwargs)
        self._alert_manager = alert_manager
        self._max_alerts = max_alerts

    def compose(self) -> ComposeResult:
        """Compose the alert inbox."""
        # Header
        with Horizontal(classes="inbox-header"):
            yield Label("ALERTS", classes="widget-header")
            yield Label("", id="alert-count", classes="alert-count")

        # Filter buttons
        with Horizontal(classes="inbox-filters"):
            yield Button("All", id="filter-all", classes="filter-btn active")
            yield Button("Trade", id="filter-trade", classes="filter-btn")
            yield Button("System", id="filter-system", classes="filter-btn")
            yield Button("Errors", id="filter-error", classes="filter-btn")

        # Alert list
        yield ListView(id="alert-list")

        # Empty state
        yield Label("No alerts", id="empty-inbox", classes="empty-inbox")

    def on_mount(self) -> None:
        """Initialize on mount."""
        self._refresh_alerts()

    def set_alert_manager(self, manager: AlertManager) -> None:
        """Set the alert manager.

        Args:
            manager: AlertManager instance.
        """
        self._alert_manager = manager
        self._refresh_alerts()

    def _refresh_alerts(self) -> None:
        """Refresh the alerts list based on current filter."""
        if self._alert_manager is None:
            self._show_empty()
            return

        # Get filtered alerts
        categories = self.category_filter if self.category_filter else None
        alerts = self._alert_manager.get_history(
            limit=self._max_alerts,
            categories=categories,
            min_severity=self.min_severity,
        )

        # Update count
        try:
            count_label = self.query_one("#alert-count", Label)
            count_label.update(f"({len(alerts)})")
        except Exception:
            pass

        # Update list
        try:
            list_view = self.query_one("#alert-list", ListView)
            empty_label = self.query_one("#empty-inbox", Label)

            list_view.clear()

            if not alerts:
                list_view.display = False
                empty_label.display = True
            else:
                list_view.display = True
                empty_label.display = False

                for alert in alerts:
                    list_view.append(AlertItem(alert))

        except Exception as e:
            logger.debug(f"Failed to refresh alerts: {e}")

    def _show_empty(self) -> None:
        """Show empty state."""
        try:
            list_view = self.query_one("#alert-list", ListView)
            empty_label = self.query_one("#empty-inbox", Label)

            list_view.clear()
            list_view.display = False
            empty_label.display = True
        except Exception:
            pass

    def _set_active_filter(self, filter_name: str) -> None:
        """Set the active filter and update button states.

        Args:
            filter_name: Name of the filter (all, trade, system, error).
        """
        self.active_filter = filter_name

        # Update button styles
        try:
            for btn_id in ["filter-all", "filter-trade", "filter-system", "filter-error"]:
                btn = self.query_one(f"#{btn_id}", Button)
                expected_id = f"filter-{filter_name}"
                if btn_id == expected_id:
                    btn.add_class("active")
                else:
                    btn.remove_class("active")
        except Exception:
            pass

    def _get_alert_category(self, name: str) -> Any:
        """Get AlertCategory enum value by name.

        Args:
            name: Category name (e.g., 'TRADE', 'SYSTEM').

        Returns:
            AlertCategory enum value.
        """
        from gpt_trader.tui.services.alert_manager import AlertCategory

        return getattr(AlertCategory, name)

    def _get_alert_severity(self, name: str) -> Any:
        """Get AlertSeverity enum value by name.

        Args:
            name: Severity name (e.g., 'WARNING', 'ERROR').

        Returns:
            AlertSeverity enum value.
        """
        from gpt_trader.tui.services.alert_manager import AlertSeverity

        return getattr(AlertSeverity, name)

    # === Actions ===

    def action_filter_all(self) -> None:
        """Show all alerts."""
        self.category_filter = set()
        self.min_severity = None
        self._set_active_filter("all")
        self._refresh_alerts()

    def action_filter_trade(self) -> None:
        """Show trade-related alerts."""
        self.category_filter = {
            self._get_alert_category("TRADE"),
            self._get_alert_category("POSITION"),
        }
        self.min_severity = None
        self._set_active_filter("trade")
        self._refresh_alerts()

    def action_filter_system(self) -> None:
        """Show system alerts."""
        self.category_filter = {
            self._get_alert_category("SYSTEM"),
            self._get_alert_category("STRATEGY"),
        }
        self.min_severity = None
        self._set_active_filter("system")
        self._refresh_alerts()

    def action_filter_error(self) -> None:
        """Show error and risk alerts."""
        self.category_filter = {
            self._get_alert_category("ERROR"),
            self._get_alert_category("RISK"),
        }
        self.min_severity = self._get_alert_severity("WARNING")
        self._set_active_filter("error")
        self._refresh_alerts()

    def action_clear_alerts(self) -> None:
        """Clear all alerts."""
        if self._alert_manager:
            self._alert_manager.clear_history()
            self._refresh_alerts()
            self.notify("Alerts cleared", timeout=2)

    # === Event Handlers ===

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle filter button clicks."""
        button_id = event.button.id

        if button_id == "filter-all":
            self.action_filter_all()
        elif button_id == "filter-trade":
            self.action_filter_trade()
        elif button_id == "filter-system":
            self.action_filter_system()
        elif button_id == "filter-error":
            self.action_filter_error()

    # === Public Methods ===

    def refresh(self) -> None:
        """Public method to refresh the alerts list."""
        self._refresh_alerts()

    def get_unread_count(self) -> int:
        """Get count of alerts matching current filter.

        Returns:
            Number of alerts in current view.
        """
        if self._alert_manager is None:
            return 0

        categories = self.category_filter if self.category_filter else None
        alerts = self._alert_manager.get_history(
            limit=100,
            categories=categories,
            min_severity=self.min_severity,
        )
        return len(alerts)
