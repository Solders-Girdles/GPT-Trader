"""Execution issues modal for displaying recent rejections and retries."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from gpt_trader.tui.responsive import calculate_modal_width

if TYPE_CHECKING:
    from gpt_trader.tui.types import ExecutionIssue, ExecutionMetrics


class ExecutionIssuesModal(ModalScreen):
    """Modal displaying recent execution issues.

    Shows:
    - Recent rejections with timestamp, symbol, side, qty, price, reason
    - Recent retries with timestamp, symbol, side, qty, price, reason

    Keyboard shortcuts:
        Escape: Close modal
    """

    BINDINGS = [
        ("escape", "dismiss", "Close"),
    ]

    def __init__(self, metrics: ExecutionMetrics) -> None:
        """Initialize execution issues modal.

        Args:
            metrics: ExecutionMetrics containing recent issues.
        """
        super().__init__()
        self.metrics = metrics

    def compose(self) -> ComposeResult:
        """Compose modal layout."""
        with Container(id="execution-issues-modal"):
            with Vertical():
                # Header
                yield Label("Execution Issues", id="exec-issues-title")

                # Rejections section
                yield Static("─── Recent Rejections ───", classes="section-header")
                if self.metrics.recent_rejections:
                    for issue in self.metrics.recent_rejections[:10]:
                        yield Static(self._format_issue_row(issue), classes="issue-row")
                    if len(self.metrics.recent_rejections) > 10:
                        more = len(self.metrics.recent_rejections) - 10
                        yield Static(f"  … and {more} more", classes="muted")
                else:
                    yield Static("  —", classes="muted")

                # Retries section
                yield Static("─── Recent Retries ───", classes="section-header")
                if self.metrics.recent_retries:
                    for issue in self.metrics.recent_retries[:10]:
                        yield Static(self._format_issue_row(issue), classes="issue-row")
                    if len(self.metrics.recent_retries) > 10:
                        more = len(self.metrics.recent_retries) - 10
                        yield Static(f"  … and {more} more", classes="muted")
                else:
                    yield Static("  —", classes="muted")

                # Summary
                yield Static("─── Summary ───", classes="section-header")
                total_rejections = (
                    self.metrics.submissions_rejected + self.metrics.submissions_failed
                )
                yield Static(
                    f"Total in window: {total_rejections} rejections, "
                    f"{self.metrics.retry_total} retries",
                    classes="summary-line",
                )

                # Close button
                yield Button("Close", variant="primary", id="close-btn")

    def on_mount(self) -> None:
        """Set dynamic width on mount."""
        width = calculate_modal_width(self.app.size.width, "medium")
        self.query_one("#execution-issues-modal").styles.width = width

    def _format_issue_row(self, issue: ExecutionIssue) -> Text:
        """Format a single issue row.

        Args:
            issue: ExecutionIssue to format.

        Returns:
            Rich Text object for display.
        """
        # Format timestamp as HH:MM:SS
        ts = time.localtime(issue.timestamp)
        time_str = f"{ts.tm_hour:02d}:{ts.tm_min:02d}:{ts.tm_sec:02d}"

        # Format price/quantity
        qty_str = f"{issue.quantity:.4f}" if issue.quantity else "—"
        price_str = f"{issue.price:.2f}" if issue.price else "—"

        # Color based on side
        side_color = "green" if issue.side.upper() == "BUY" else "red"

        # Build the row: HH:MM:SS SYMBOL SIDE QTY @ PRICE — REASON
        reason_text = issue.reason or "unknown"
        if issue.reason_detail:
            reason_text = f"{reason_text}:{issue.reason_detail}"

        parts = [
            "  ",
            Text(time_str, style="dim"),
            " ",
            Text(issue.symbol or "—", style="bold"),
            " ",
            Text(issue.side.upper() if issue.side else "—", style=side_color),
            " ",
            Text(qty_str, style="cyan"),
            " @ ",
            Text(price_str, style="cyan"),
            " — ",
            Text(reason_text, style="yellow"),
        ]

        return Text.assemble(*parts)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "close-btn":
            self.dismiss()

    async def action_dismiss(self) -> None:
        """Dismiss the modal."""
        self.dismiss()
