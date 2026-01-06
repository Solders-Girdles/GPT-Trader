"""
Onboarding Checklist Widget.

Displays setup progress as a visual checklist for first-run guidance.
Shows completion status for each step and highlights the next action.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.services.onboarding_service import (
    ChecklistItem,
    OnboardingStatus,
    get_onboarding_service,
)
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class OnboardingChecklist(Static):
    """Widget showing setup progress as a checklist.

    Displays each onboarding step with completion indicators:
    - ✓ Completed items (green)
    - ○ Pending required items (yellow, highlighted if next)
    - · Optional items (dim)
    """

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def __init__(
        self,
        compact: bool = False,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the checklist widget.

        Args:
            compact: If True, show single-line summary instead of full list.
            id: Widget ID.
            classes: CSS classes.
        """
        super().__init__(id=id, classes=classes)
        self._compact = compact
        self._onboarding = get_onboarding_service()

    def compose(self) -> ComposeResult:
        yield Label("SETUP PROGRESS", classes="widget-header")
        with Vertical(id="checklist-items", classes="checklist-items"):
            # Items are added dynamically in update_checklist
            pass
        yield Label("", id="checklist-summary", classes="checklist-summary")

    def on_mount(self) -> None:
        # Register with state registry for state updates
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from state registry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update checklist automatically."""
        if state is None:
            return

        status = self._onboarding.get_status(state)
        self.update_checklist(status)

    def on_state_updated(self, state: TuiState) -> None:
        """Called by StateRegistry when state changes."""
        self.state = state

    def update_checklist(self, status: OnboardingStatus) -> None:
        """Update the checklist display from status.

        Args:
            status: Current onboarding status.
        """
        try:
            items_container = self.query_one("#checklist-items", Vertical)
            summary_label = self.query_one("#checklist-summary", Label)

            # Clear existing items
            items_container.remove_children()

            if self._compact:
                # Compact mode: show progress bar style
                self._update_compact(status, summary_label)
                items_container.display = False
            else:
                # Full mode: show each item
                items_container.display = True
                next_step = status.get_next_step()

                for item in status.items:
                    item_label = self._make_item_label(item, is_next=item == next_step)
                    items_container.mount(item_label)

                # Update summary
                if status.is_ready:
                    summary_label.update("[green]✓ Ready to trade[/green]")
                else:
                    next_action = next_step.label if next_step else "Complete setup"
                    summary_label.update(f"[dim]Next: {next_action}[/dim]")

        except Exception as e:
            logger.debug("Failed updating checklist: %s", e)

    def _make_item_label(self, item: ChecklistItem, is_next: bool = False) -> Label:
        """Create a label for a checklist item.

        Args:
            item: The checklist item.
            is_next: Whether this is the next step to complete.

        Returns:
            Label widget for the item.
        """
        if item.completed:
            # Completed - green checkmark
            icon = "[green]✓[/green]"
            text = f"{icon} {item.label}"
            css_class = "checklist-item completed"
        elif not item.required:
            # Optional - dim dot
            icon = "[dim]·[/dim]"
            text = f"{icon} {item.label} [dim](optional)[/dim]"
            css_class = "checklist-item optional"
        elif is_next:
            # Next required step - highlighted
            icon = "[yellow]→[/yellow]"
            text = f"{icon} [bold]{item.label}[/bold]"
            css_class = "checklist-item next"
        else:
            # Pending required - empty circle
            icon = "[yellow]○[/yellow]"
            text = f"{icon} {item.label}"
            css_class = "checklist-item pending"

        label = Label(text, classes=css_class)
        return label

    def _update_compact(self, status: OnboardingStatus, label: Label) -> None:
        """Update compact mode display.

        Args:
            status: Current onboarding status.
            label: Label to update with compact summary.
        """
        if status.is_ready:
            label.update("[green]✓ Ready[/green]")
        else:
            # Progress bar style: ●●○○
            filled = "●" * status.required_completed
            empty = "○" * (status.required_count - status.required_completed)
            progress = f"[green]{filled}[/green][dim]{empty}[/dim]"
            label.update(f"Setup: {progress} {status.ready_label}")


__all__ = ["OnboardingChecklist"]
