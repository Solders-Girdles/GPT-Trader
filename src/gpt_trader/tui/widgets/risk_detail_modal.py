"""Risk detail modal for displaying comprehensive risk status."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from gpt_trader.tui.responsive import calculate_modal_width
from gpt_trader.tui.thresholds import (
    DEFAULT_RISK_THRESHOLDS,
    StatusLevel,
    get_loss_ratio_status,
    get_risk_score_status,
    get_risk_status_label,
    get_status_color,
)

if TYPE_CHECKING:
    from gpt_trader.tui.types import RiskGuard, RiskState


class RiskDetailModal(ModalScreen):
    """Modal displaying detailed risk management status.

    Shows:
    - Overall risk status with score breakdown
    - Daily loss utilization (current vs limit)
    - Reduce-only mode status and reason
    - Max leverage setting
    - Full list of active guards with color coding
    """

    BINDINGS = [("escape", "dismiss", "Close")]

    def __init__(self, risk_data: RiskState) -> None:
        """Initialize risk detail modal.

        Args:
            risk_data: RiskState containing current risk metrics.
        """
        super().__init__()
        self.risk_data = risk_data

    def compose(self) -> ComposeResult:
        """Compose modal layout."""
        data = self.risk_data

        # Calculate risk score for display
        risk_score = self._calculate_risk_score(data)
        risk_status = get_risk_score_status(risk_score, DEFAULT_RISK_THRESHOLDS)
        risk_label = get_risk_status_label(risk_status)
        risk_color = get_status_color(risk_status)

        # Calculate loss utilization
        loss_status = get_loss_ratio_status(data.current_daily_loss_pct, data.daily_loss_limit_pct)
        loss_color = get_status_color(loss_status)

        with Container(id="risk-detail-modal"):
            with Vertical():
                # Header
                yield Label("Risk Management Details", id="risk-detail-title")

                # Overall status section
                yield Static("─── Overall Status ───", classes="section-header")
                yield Static(
                    Text.assemble(
                        "Risk Level: ",
                        Text(risk_label, style=f"{risk_color} bold"),
                        f" (score: {risk_score})",
                    )
                )

                # Daily loss section
                yield Static("─── Daily P&L ───", classes="section-header")
                if data.daily_loss_limit_pct > 0:
                    utilization = (
                        abs(data.current_daily_loss_pct) / data.daily_loss_limit_pct
                    ) * 100
                    utilization_text = Text(f"{utilization:.1f}%", style=loss_color)

                    if data.current_daily_loss_pct < 0:
                        pnl_text = Text(f"{data.current_daily_loss_pct:.2%}", style="red")
                        pnl_type = "(loss)"
                    elif data.current_daily_loss_pct > 0:
                        pnl_text = Text(f"+{data.current_daily_loss_pct:.2%}", style="green")
                        pnl_type = "(profit)"
                    else:
                        pnl_text = Text("0.00%", style="dim")
                        pnl_type = ""

                    yield Static(Text.assemble("Current P&L: ", pnl_text, f" {pnl_type}"))
                    yield Static(f"Daily Limit: {data.daily_loss_limit_pct:.2%}")
                    yield Static(Text.assemble("Utilization: ", utilization_text, " of limit"))
                else:
                    yield Static("Daily Limit: Not configured", classes="muted")

                # Leverage section
                yield Static("─── Leverage ───", classes="section-header")
                yield Static(f"Max Leverage: {data.max_leverage}x")

                # Reduce-only section
                yield Static("─── Reduce-Only Mode ───", classes="section-header")
                if data.reduce_only_mode:
                    yield Static(Text.assemble("Status: ", Text("ACTIVE", style="red bold")))
                    reason = data.reduce_only_reason or "Risk limit exceeded"
                    yield Static(f"Reason: {reason}")
                else:
                    yield Static(Text.assemble("Status: ", Text("Inactive", style="green")))

                # Active guards section with enhanced display
                yield Static("─── Active Guards ───", classes="section-header")

                # Use enhanced guards if available, fall back to legacy active_guards
                guards_to_display = self._get_sorted_guards(data)

                if guards_to_display:
                    count = len(guards_to_display)
                    if count >= 3:
                        guard_status = StatusLevel.CRITICAL
                    else:
                        guard_status = StatusLevel.WARNING
                    guard_color = get_status_color(guard_status)

                    yield Static(
                        Text.assemble(Text(f"{count} guard(s) active", style=f"{guard_color} bold"))
                    )

                    # Display each guard with enhanced info
                    for guard_info in guards_to_display:
                        yield Static(self._format_guard_row(guard_info))
                else:
                    yield Static(Text("No active guards", style="dim"), classes="muted")

                # Score breakdown section
                yield Static("─── Score Breakdown ───", classes="section-header")
                yield Static(self._format_score_breakdown(data), classes="muted")

                # Close button
                yield Button("Close", variant="primary", id="close-btn")

    def on_mount(self) -> None:
        """Set dynamic width on mount."""
        width = calculate_modal_width(self.app.size.width, "medium")
        self.query_one("#risk-detail-modal").styles.width = width

    def _calculate_risk_score(self, data: RiskState) -> int:
        """Calculate overall risk score.

        Mirrors the logic in RiskWidget._update_risk_status().
        """
        score = 0

        # Factor 1: Daily loss usage
        loss_status = get_loss_ratio_status(
            data.current_daily_loss_pct,
            data.daily_loss_limit_pct,
            DEFAULT_RISK_THRESHOLDS,
        )
        if loss_status == StatusLevel.CRITICAL:
            score += 3
        elif loss_status == StatusLevel.WARNING:
            score += 2
        elif data.daily_loss_limit_pct > 0:
            loss_ratio = abs(data.current_daily_loss_pct) / data.daily_loss_limit_pct
            if loss_ratio >= 0.25:
                score += 1

        # Factor 2: Reduce-only mode
        if data.reduce_only_mode:
            score += 3

        # Factor 3: Active guards
        if len(data.active_guards) >= 3:
            score += 2
        elif len(data.active_guards) >= 1:
            score += 1

        return score

    def _format_score_breakdown(self, data: RiskState) -> str:
        """Format score breakdown for display."""
        lines = []

        # Loss contribution
        loss_status = get_loss_ratio_status(
            data.current_daily_loss_pct,
            data.daily_loss_limit_pct,
            DEFAULT_RISK_THRESHOLDS,
        )
        if loss_status == StatusLevel.CRITICAL:
            lines.append("Loss >75% of limit: +3")
        elif loss_status == StatusLevel.WARNING:
            lines.append("Loss 50-75% of limit: +2")
        elif data.daily_loss_limit_pct > 0:
            loss_ratio = abs(data.current_daily_loss_pct) / data.daily_loss_limit_pct
            if loss_ratio >= 0.25:
                lines.append("Loss 25-50% of limit: +1")

        # Reduce-only contribution
        if data.reduce_only_mode:
            lines.append("Reduce-only active: +3")

        # Guards contribution
        guard_count = len(data.active_guards)
        if guard_count >= 3:
            lines.append(f"{guard_count} guards active: +2")
        elif guard_count >= 1:
            lines.append(f"{guard_count} guard(s) active: +1")

        if not lines:
            return "No risk factors detected"

        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "close-btn":
            self.dismiss()

    async def action_dismiss(self) -> None:
        """Dismiss the modal."""
        self.dismiss()

    def _get_sorted_guards(self, data: RiskState) -> list[RiskGuard]:
        """Get guards sorted by severity (highest first).

        Falls back to legacy active_guards if enhanced guards not available.

        Args:
            data: RiskState containing guard information.

        Returns:
            List of RiskGuard objects sorted by severity.
        """
        from gpt_trader.tui.types import RiskGuard

        # Prefer enhanced guards list
        if data.guards:
            # Sort by severity (descending) then by name
            return sorted(
                data.guards,
                key=lambda g: (-g.severity_order, g.name),
            )

        # Fall back to legacy active_guards (convert to RiskGuard objects)
        if data.active_guards:
            return [
                RiskGuard(
                    name=guard_name,
                    severity=self._infer_severity(guard_name),
                )
                for guard_name in data.active_guards
            ]

        return []

    def _infer_severity(self, guard_name: str) -> str:
        """Infer severity from guard name for legacy guards.

        Args:
            guard_name: Name of the guard.

        Returns:
            Inferred severity level.
        """
        name_lower = guard_name.lower()

        # Critical guards
        if any(word in name_lower for word in ["drawdown", "loss", "margin", "liquidation"]):
            return "CRITICAL"

        # High severity guards
        if any(word in name_lower for word in ["volatility", "exposure", "leverage"]):
            return "HIGH"

        # Medium severity guards
        if any(word in name_lower for word in ["position", "size", "daily"]):
            return "MEDIUM"

        # Default to LOW
        return "LOW"

    def _format_guard_row(self, guard: RiskGuard) -> Text:
        """Format a single guard row with severity and timestamp.

        Args:
            guard: RiskGuard to format.

        Returns:
            Rich Text object for display.
        """
        # Severity color mapping
        severity_colors = {
            "CRITICAL": "red",
            "HIGH": "yellow",
            "MEDIUM": "cyan",
            "LOW": "dim",
        }
        severity_color = severity_colors.get(guard.severity.upper(), "dim")

        # Severity badge
        severity_badge = f"[{guard.severity[:1]}]"  # C, H, M, L

        # Format last triggered time
        if guard.last_triggered > 0:
            age_str = self._format_age(guard.last_triggered)
            # Color based on recency
            now = time.time()
            age_seconds = now - guard.last_triggered
            if age_seconds < 60:
                age_color = "red bold"  # Very recent
            elif age_seconds < 300:
                age_color = "yellow"  # Within 5 min
            elif age_seconds < 3600:
                age_color = "cyan"  # Within 1 hour
            else:
                age_color = "dim"  # Old
            triggered_text = Text(f"({age_str})", style=age_color)
        else:
            triggered_text = Text("(never)", style="dim")

        # Build the row
        parts = [
            "  • ",
            Text(severity_badge, style=severity_color),
            " ",
            Text(guard.name, style=severity_color),
            " ",
            triggered_text,
        ]

        # Add trigger count if > 0
        if guard.triggered_count > 0:
            parts.append(Text(f" ×{guard.triggered_count}", style="dim"))

        return Text.assemble(*parts)

    def _format_age(self, timestamp: float) -> str:
        """Format a timestamp as relative age.

        Args:
            timestamp: Epoch timestamp.

        Returns:
            Human-readable age string (e.g., "45s ago", "3m ago", "2h ago").
        """
        now = time.time()
        diff = now - timestamp

        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            return f"{int(diff / 60)}m ago"
        elif diff < 86400:
            return f"{int(diff / 3600)}h ago"
        else:
            return f"{int(diff / 86400)}d ago"
