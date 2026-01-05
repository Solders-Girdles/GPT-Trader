from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import Label, ProgressBar, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.staleness_helpers import get_staleness_banner
from gpt_trader.tui.thresholds import (
    DEFAULT_RISK_THRESHOLDS,
    get_loss_ratio_status,
    get_risk_score_status,
    get_risk_status_label,
    get_status_class,
)
from gpt_trader.tui.types import RiskState
from gpt_trader.tui.widgets.tile_states import TileBanner
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class RiskWidget(Static):
    """Widget to display risk management status with visual indicators.

    Features:
    - Progress bars for daily loss and leverage utilization
    - Color-coded risk status (low/medium/high)
    - Reduce-only mode indicator
    - Active risk guards display
    """

    # Styles moved to styles/widgets/risk.tcss

    def compose(self) -> ComposeResult:
        yield Label("RISK MANAGEMENT", classes="widget-header")

        # Staleness/degraded banner (initially hidden)
        yield TileBanner()

        with Vertical(classes="risk-section"):
            # Daily Loss Progress
            yield Label("Daily P&L:", classes="risk-label")
            yield ProgressBar(id="daily-loss-bar", total=100, show_eta=False)
            yield Label("-", id="daily-loss-value", classes="risk-value")

        with Grid(classes="risk-grid"):
            # Max Leverage
            yield Label("Max Leverage:", classes="risk-label")
            yield Label("-", id="max-leverage", classes="risk-value")

            # Risk Status
            yield Label("Risk Status:", classes="risk-label")
            yield Label("LOW", id="risk-status", classes="risk-value risk-status-low")

            # Reduce Only Mode
            yield Label("Reduce Only:", classes="risk-label")
            yield Label("OFF", id="reduce-only", classes="risk-value")

            # Active Guards
            yield Label("Active Guards:", classes="risk-label")
            yield Label("None", id="active-guards", classes="risk-value")

    @safe_update
    def update_risk(self, data: RiskState, state: TuiState | None = None) -> None:
        """Update risk display with current data.

        Args:
            data: RiskState containing current risk metrics.
            state: Optional TuiState for staleness banner updates.
        """
        # Update staleness banner if state provided
        if state is not None:
            self._update_staleness_banner(state)

        # Update Max Leverage
        self.query_one("#max-leverage", Label).update(f"{data.max_leverage}x")

        # Calculate and update daily loss progress
        self._update_daily_loss(data)

        # Determine and update risk status
        self._update_risk_status(data)

        # Update Reduce Only
        self._update_reduce_only(data)

        # Update Active Guards
        self._update_guards(data)

    def _update_staleness_banner(self, state: TuiState) -> None:
        """Update staleness/degraded banner based on state."""
        try:
            banner = self.query_one(TileBanner)
            staleness_result = get_staleness_banner(state)
            if staleness_result:
                banner.update_banner(staleness_result[0], severity=staleness_result[1])
            else:
                banner.hide()
        except Exception:
            pass

    def _update_daily_loss(self, data: RiskState) -> None:
        """Update daily loss progress bar and label using shared thresholds."""
        progress_bar = self.query_one("#daily-loss-bar", ProgressBar)
        value_label = self.query_one("#daily-loss-value", Label)

        if data.daily_loss_limit_pct > 0:
            # Calculate percentage of limit used (using abs for correct handling)
            current_loss = abs(data.current_daily_loss_pct)
            limit = data.daily_loss_limit_pct
            pct_used = min((current_loss / limit) * 100, 100)

            progress_bar.update(progress=pct_used)

            # Format the value display
            if data.current_daily_loss_pct < 0:
                # Loss (negative P&L)
                value_text = Text(f"{data.current_daily_loss_pct:.2%} ", style="red")
                value_text.append(f"of {limit:.1%} limit", style="dim")
            elif data.current_daily_loss_pct > 0:
                # Profit (positive P&L)
                value_text = Text(f"+{data.current_daily_loss_pct:.2%} ", style="green")
                value_text.append("(profit)", style="dim")
            else:
                value_text = Text("0.00% ", style="dim")
                value_text.append(f"of {limit:.1%} limit", style="dim")

            value_label.update(value_text)

            # Color the progress bar using shared status thresholds
            loss_status = get_loss_ratio_status(
                data.current_daily_loss_pct, data.daily_loss_limit_pct
            )
            status_class = get_status_class(loss_status)

            # Remove old classes and add new
            progress_bar.remove_class("risk-status-high", "risk-status-medium", "risk-status-low")
            progress_bar.remove_class("status-ok", "status-warning", "status-critical")
            progress_bar.add_class(status_class)
        else:
            progress_bar.update(progress=0)
            value_label.update("No limit configured")

    def _update_risk_status(self, data: RiskState) -> None:
        """Calculate and update overall risk status using shared thresholds."""
        status_label = self.query_one("#risk-status", Label)

        # Determine risk level based on multiple factors
        risk_score = 0

        # Factor 1: Daily loss usage (using shared threshold)
        loss_status = get_loss_ratio_status(
            data.current_daily_loss_pct,
            data.daily_loss_limit_pct,
            DEFAULT_RISK_THRESHOLDS,
        )
        if loss_status.value == "critical":
            risk_score += 3
        elif loss_status.value == "warning":
            risk_score += 2
        elif data.daily_loss_limit_pct > 0:
            # Check for low usage (25-50% = 1 point)
            loss_ratio = abs(data.current_daily_loss_pct) / data.daily_loss_limit_pct
            if loss_ratio >= 0.25:
                risk_score += 1

        # Factor 2: Reduce-only mode
        if data.reduce_only_mode:
            risk_score += 3

        # Factor 3: Active guards
        if len(data.active_guards) >= 3:
            risk_score += 2
        elif len(data.active_guards) >= 1:
            risk_score += 1

        # Factor 4: High leverage per position
        # Note: position_leverage removed as GPT-Trader focuses on spot trading
        # For perpetuals/margin, this check would need to be re-added

        # Determine status from score using shared thresholds
        status = get_risk_score_status(risk_score, DEFAULT_RISK_THRESHOLDS)
        status_text = get_risk_status_label(status)
        status_class = get_status_class(status)

        status_label.update(status_text)
        status_label.remove_class("risk-status-low", "risk-status-medium", "risk-status-high")
        status_label.remove_class("status-ok", "status-warning", "status-critical")
        status_label.add_class(status_class)

    def _update_reduce_only(self, data: RiskState) -> None:
        """Update reduce-only mode display."""
        reduce_only_label = self.query_one("#reduce-only", Label)
        if data.reduce_only_mode:
            reason = data.reduce_only_reason or "Risk limit"
            # Truncate long reasons
            if len(reason) > 20:
                reason = reason[:17] + "..."
            reduce_only_label.update(Text(f"ON ({reason})", style="red bold"))
            reduce_only_label.add_class("risk-alert")
        else:
            reduce_only_label.update(Text("OFF", style="green"))
            reduce_only_label.remove_class("risk-alert")

    def _update_guards(self, data: RiskState) -> None:
        """Update active guards display."""
        guards_label = self.query_one("#active-guards", Label)
        if data.active_guards:
            # Show count and first few guards
            count = len(data.active_guards)
            if count <= 2:
                guards_label.update(", ".join(data.active_guards))
            else:
                first_two = ", ".join(data.active_guards[:2])
                guards_label.update(f"{first_two} +{count - 2} more")
        else:
            guards_label.update(Text("None", style="dim"))
