from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Grid, Vertical
from textual.widgets import Label, ProgressBar, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import RiskState
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class RiskWidget(Static):
    """Widget to display risk management status with visual indicators.

    Features:
    - Progress bars for daily loss and leverage utilization
    - Color-coded risk status (low/medium/high)
    - Reduce-only mode indicator
    - Active risk guards display
    """

    DEFAULT_CSS = """
    RiskWidget {
        layout: vertical;
        height: auto;
    }

    RiskWidget .risk-section {
        margin-bottom: 1;
    }

    RiskWidget ProgressBar {
        margin: 0 1;
    }

    RiskWidget .risk-status-low {
        color: $success;
    }

    RiskWidget .risk-status-medium {
        color: $warning;
    }

    RiskWidget .risk-status-high {
        color: $error;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("RISK MANAGEMENT", classes="header")

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
    def update_risk(self, data: RiskState) -> None:
        """Update risk display with current data.

        Args:
            data: RiskState containing current risk metrics.
        """
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

    def _update_daily_loss(self, data: RiskState) -> None:
        """Update daily loss progress bar and label."""
        progress_bar = self.query_one("#daily-loss-bar", ProgressBar)
        value_label = self.query_one("#daily-loss-value", Label)

        if data.daily_loss_limit_pct > 0:
            # Calculate percentage of limit used
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

            # Color the progress bar based on usage
            if pct_used >= 75:
                progress_bar.add_class("risk-status-high")
                progress_bar.remove_class("risk-status-medium", "risk-status-low")
            elif pct_used >= 50:
                progress_bar.add_class("risk-status-medium")
                progress_bar.remove_class("risk-status-high", "risk-status-low")
            else:
                progress_bar.add_class("risk-status-low")
                progress_bar.remove_class("risk-status-high", "risk-status-medium")
        else:
            progress_bar.update(progress=0)
            value_label.update("No limit configured")

    def _update_risk_status(self, data: RiskState) -> None:
        """Calculate and update overall risk status."""
        status_label = self.query_one("#risk-status", Label)

        # Determine risk level based on multiple factors
        risk_score = 0

        # Factor 1: Daily loss usage
        if data.daily_loss_limit_pct > 0:
            loss_pct = abs(data.current_daily_loss_pct) / data.daily_loss_limit_pct
            if loss_pct >= 0.75:
                risk_score += 3
            elif loss_pct >= 0.50:
                risk_score += 2
            elif loss_pct >= 0.25:
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

        # Determine status from score
        if risk_score >= 5:
            status_label.update("HIGH")
            status_label.remove_class("risk-status-low", "risk-status-medium")
            status_label.add_class("risk-status-high")
        elif risk_score >= 2:
            status_label.update("MEDIUM")
            status_label.remove_class("risk-status-low", "risk-status-high")
            status_label.add_class("risk-status-medium")
        else:
            status_label.update("LOW")
            status_label.remove_class("risk-status-medium", "risk-status-high")
            status_label.add_class("risk-status-low")

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
