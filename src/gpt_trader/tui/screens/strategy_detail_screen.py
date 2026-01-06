"""Strategy detail screen for comprehensive strategy analysis.

Provides a full-page deep dive into:
- Current strategy configuration
- Performance metrics over multiple time windows
- Recent decisions with full signal breakdown
- Per-symbol performance breakdown
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import (
    IndicatorContribution,
    RegimeData,
    StrategyPerformance,
)

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


class StrategyDetailScreen(Screen):
    """Full-page strategy analysis screen.

    Layout:
    ┌─ STRATEGY DETAIL ──────────────────────────────────────────┐
    │ Active: ensemble (v2.1)              Regime: [BULL ↑] 82%  │
    ├─ CONFIGURATION ────────────────────────────────────────────┤
    │ Buy Threshold: 0.20    Sell Threshold: -0.20               │
    │ Stop Loss: 2.0%        Take Profit: 5.0%                   │
    ├─ PERFORMANCE ──────────────────────────────────────────────┤
    │ Win Rate:  65.2%       Trades:   23                        │
    │ Return:   +12.4%       Sharpe:   1.24                      │
    │ Drawdown: -3.2%        W/L:      15/8                      │
    ├─ RECENT DECISIONS ─────────────────────────────────────────┤
    │ [DataTable of decisions with signal breakdown]             │
    └────────────────────────────────────────────────────────────┘
    """

    BINDINGS = [
        Binding("escape", "pop_screen", "Back", show=True),
        Binding("q", "pop_screen", "Back", show=False),
    ]

    DEFAULT_CSS = """
    StrategyDetailScreen {
        background: $surface;
    }

    StrategyDetailScreen #strategy-detail-container {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }

    StrategyDetailScreen .section-header {
        width: 100%;
        height: 1;
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
        margin-top: 1;
    }

    StrategyDetailScreen .section-header:first-child {
        margin-top: 0;
    }

    StrategyDetailScreen #header-section {
        height: 3;
        width: 100%;
        layout: horizontal;
    }

    StrategyDetailScreen #header-left {
        width: 1fr;
    }

    StrategyDetailScreen #header-right {
        width: auto;
    }

    StrategyDetailScreen .strategy-title {
        text-style: bold;
        color: $text;
    }

    StrategyDetailScreen .strategy-subtitle {
        color: $text-muted;
    }

    StrategyDetailScreen .regime-badge {
        text-style: bold;
        padding: 0 1;
    }

    StrategyDetailScreen .regime-badge.bullish {
        color: $success;
    }

    StrategyDetailScreen .regime-badge.bearish {
        color: $error;
    }

    StrategyDetailScreen .regime-badge.sideways {
        color: $warning;
    }

    StrategyDetailScreen .regime-badge.crisis {
        color: $error;
        text-style: bold reverse;
    }

    StrategyDetailScreen .config-grid {
        height: 3;
        layout: grid;
        grid-size: 4 2;
        grid-columns: 1fr 1fr 1fr 1fr;
        padding: 0 1;
    }

    StrategyDetailScreen .config-label {
        color: $text-muted;
    }

    StrategyDetailScreen .config-value {
        text-style: bold;
    }

    StrategyDetailScreen .perf-grid {
        height: 4;
        layout: grid;
        grid-size: 4 3;
        grid-columns: 1fr 1fr 1fr 1fr;
        padding: 0 1;
    }

    StrategyDetailScreen .perf-label {
        color: $text-muted;
    }

    StrategyDetailScreen .perf-value {
        text-style: bold;
    }

    StrategyDetailScreen .perf-value.positive {
        color: $success;
    }

    StrategyDetailScreen .perf-value.negative {
        color: $error;
    }

    StrategyDetailScreen #decisions-section {
        height: 1fr;
        min-height: 10;
    }

    StrategyDetailScreen #decisions-table {
        height: 100%;
    }

    StrategyDetailScreen #signal-detail {
        height: auto;
        min-height: 5;
        max-height: 8;
        background: $surface;
        border: tall $primary;
        padding: 0 1;
        margin-top: 1;
    }

    StrategyDetailScreen .signal-detail-header {
        color: $text-muted;
        text-style: italic;
        height: 1;
    }

    StrategyDetailScreen .signal-bar-row {
        height: 1;
    }

    StrategyDetailScreen .empty-state {
        color: $text-muted;
        text-align: center;
        padding: 2;
    }
    """

    def __init__(self, **kwargs) -> None:
        """Initialize the strategy detail screen."""
        super().__init__(**kwargs)
        self._selected_symbol: str | None = None

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="strategy-detail-container"):
            # Header section
            yield Label("STRATEGY DETAIL", classes="section-header")
            with Horizontal(id="header-section"):
                with Vertical(id="header-left"):
                    yield Label("Active Strategy: --", id="strategy-name", classes="strategy-title")
                    yield Label(
                        "No active strategies", id="strategy-info", classes="strategy-subtitle"
                    )
                with Vertical(id="header-right"):
                    yield Label("--", id="regime-badge", classes="regime-badge")

            # Configuration section
            yield Label("CONFIGURATION", classes="section-header")
            with Container(classes="config-grid"):
                yield Label("Buy Threshold:", classes="config-label")
                yield Label("--", id="config-buy-threshold", classes="config-value")
                yield Label("Sell Threshold:", classes="config-label")
                yield Label("--", id="config-sell-threshold", classes="config-value")
                yield Label("Stop Loss:", classes="config-label")
                yield Label("--", id="config-stop-loss", classes="config-value")
                yield Label("Take Profit:", classes="config-label")
                yield Label("--", id="config-take-profit", classes="config-value")

            # Performance section
            yield Label("PERFORMANCE", classes="section-header")
            with Container(classes="perf-grid"):
                yield Label("Win Rate:", classes="perf-label")
                yield Label("--", id="perf-win-rate", classes="perf-value")
                yield Label("Trades:", classes="perf-label")
                yield Label("--", id="perf-trades", classes="perf-value")

                yield Label("Return:", classes="perf-label")
                yield Label("--", id="perf-return", classes="perf-value")
                yield Label("Sharpe:", classes="perf-label")
                yield Label("--", id="perf-sharpe", classes="perf-value")

                yield Label("Drawdown:", classes="perf-label")
                yield Label("--", id="perf-drawdown", classes="perf-value")
                yield Label("W/L:", classes="perf-label")
                yield Label("--", id="perf-wl", classes="perf-value")

            # Recent decisions section
            yield Label("RECENT DECISIONS", classes="section-header")
            with Container(id="decisions-section"):
                table = DataTable(id="decisions-table", zebra_stripes=True)
                table.cursor_type = "row"
                yield table

            # Signal detail panel (shown when row selected)
            with Vertical(id="signal-detail", classes="hidden"):
                yield Label("Signal Details", classes="breakdown-title")
                yield Label(
                    "Signal     Value   Wt   Contrib",
                    classes="signal-detail-header",
                )
                yield Vertical(id="signal-bars-detail")

    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        # Set up decisions table
        table = self.query_one("#decisions-table", DataTable)
        table.add_columns("Symbol", "Action", "Confidence", "Reason", "Time")

        # Register with state registry
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

        # Initial update from current state
        if hasattr(self.app, "tui_state"):
            self.on_state_updated(self.app.tui_state)

    def on_unmount(self) -> None:
        """Clean up when screen is removed."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Update the screen when state changes."""
        self._update_header(state)
        self._update_performance(state.strategy_performance)
        self._update_regime(state.regime_data)
        self._update_decisions(state)

    @safe_update
    def _update_header(self, state: TuiState) -> None:
        """Update the header section."""
        strategy_data = state.strategy_data

        # Strategy name
        name_label = self.query_one("#strategy-name", Label)
        if strategy_data.active_strategies:
            name_label.update(f"Active Strategy: {', '.join(strategy_data.active_strategies)}")
        else:
            name_label.update("Active Strategy: None")

        # Strategy info
        info_label = self.query_one("#strategy-info", Label)
        decision_count = len(strategy_data.last_decisions)
        info_label.update(f"{decision_count} decisions tracked")

    @safe_update
    def _update_performance(self, performance: StrategyPerformance) -> None:
        """Update the performance section."""
        # Win Rate
        win_rate = self.query_one("#perf-win-rate", Label)
        if performance.total_trades > 0:
            win_rate.update(f"{performance.win_rate_pct:.1f}%")
            win_rate.remove_class("positive", "negative")
            win_rate.add_class("positive" if performance.win_rate >= 0.5 else "negative")
        else:
            win_rate.update("--")
            win_rate.remove_class("positive", "negative")

        # Trades
        trades = self.query_one("#perf-trades", Label)
        trades.update(str(performance.total_trades))

        # Return
        ret = self.query_one("#perf-return", Label)
        if performance.total_return_pct != 0:
            sign = "+" if performance.total_return_pct > 0 else ""
            ret.update(f"{sign}{performance.total_return_pct:.1f}%")
            ret.remove_class("positive", "negative")
            ret.add_class("positive" if performance.total_return_pct > 0 else "negative")
        else:
            ret.update("--")
            ret.remove_class("positive", "negative")

        # Sharpe
        sharpe = self.query_one("#perf-sharpe", Label)
        if performance.sharpe_ratio != 0:
            sharpe.update(f"{performance.sharpe_ratio:.2f}")
        else:
            sharpe.update("--")

        # Drawdown
        dd = self.query_one("#perf-drawdown", Label)
        if performance.max_drawdown_pct != 0:
            dd.update(f"{performance.max_drawdown_pct:.1f}%")
            dd.remove_class("positive", "negative")
            dd.add_class("negative")
        else:
            dd.update("--")
            dd.remove_class("positive", "negative")

        # W/L
        wl = self.query_one("#perf-wl", Label)
        wl.update(f"{performance.winning_trades}/{performance.losing_trades}")

    @safe_update
    def _update_regime(self, regime: RegimeData) -> None:
        """Update the regime badge."""
        badge = self.query_one("#regime-badge", Label)

        if regime.regime != "UNKNOWN":
            badge.update(f"[{regime.short_label} {regime.icon}] {regime.confidence*100:.0f}%")
            badge.remove_class("bullish", "bearish", "sideways", "crisis")
            if regime.is_crisis:
                badge.add_class("crisis")
            elif regime.is_bullish:
                badge.add_class("bullish")
            elif regime.is_bearish:
                badge.add_class("bearish")
            else:
                badge.add_class("sideways")
        else:
            badge.update("--")
            badge.remove_class("bullish", "bearish", "sideways", "crisis")

    @safe_update
    def _update_decisions(self, state: TuiState) -> None:
        """Update the decisions table."""
        table = self.query_one("#decisions-table", DataTable)
        strategy_data = state.strategy_data

        # Clear and rebuild table
        table.clear()

        if not strategy_data.last_decisions:
            return

        for symbol, decision in strategy_data.last_decisions.items():
            action = decision.action.upper()

            # Color code action
            if action == "BUY":
                formatted_action = f"[green]{action}[/green]"
            elif action == "SELL":
                formatted_action = f"[red]{action}[/red]"
            else:
                formatted_action = f"[yellow]{action}[/yellow]"

            # Format confidence
            confidence = f"{decision.confidence:.2f}"

            # Format timestamp
            time_str = ""
            if decision.timestamp > 0:
                time_str = datetime.fromtimestamp(decision.timestamp).strftime("%H:%M:%S")

            table.add_row(
                symbol, formatted_action, confidence, decision.reason, time_str, key=symbol
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlight to show signal breakdown."""
        if event.row_key is None:
            self._hide_signal_detail()
            return

        symbol = str(event.row_key.value)
        self._selected_symbol = symbol
        self._show_signal_detail(symbol)

    def _show_signal_detail(self, symbol: str) -> None:
        """Show signal breakdown for the selected decision."""
        if not hasattr(self.app, "tui_state"):
            return

        state = self.app.tui_state
        decision = state.strategy_data.last_decisions.get(symbol)

        if not decision or not decision.contributions:
            self._hide_signal_detail()
            return

        try:
            # Show the panel
            panel = self.query_one("#signal-detail", Vertical)
            panel.remove_class("hidden")

            # Update signal bars
            bars = self.query_one("#signal-bars-detail", Vertical)
            bars.remove_children()

            sorted_contributions = sorted(
                decision.contributions,
                key=lambda c: abs(c.contribution),
                reverse=True,
            )

            for contrib in sorted_contributions[:6]:
                row = self._format_signal_detail_row(contrib)
                bars.mount(row)

        except Exception:
            self._hide_signal_detail()

    def _hide_signal_detail(self) -> None:
        """Hide the signal detail panel."""
        try:
            panel = self.query_one("#signal-detail", Vertical)
            panel.add_class("hidden")
        except Exception:
            pass

    def _format_signal_detail_row(self, contrib: IndicatorContribution) -> Static:
        """Format a signal detail row with name, value, weight, bar, and direction.

        Args:
            contrib: IndicatorContribution with signal data.

        Returns:
            Static widget with formatted row.
        """
        content = self._build_signal_detail_content(contrib)
        return Static(content, classes="signal-bar-row")

    def _build_signal_detail_content(self, contrib: IndicatorContribution) -> str:
        """Build the markup string for a signal detail row.

        Format: Signal     Value   Wt   Contrib
                RSI       35.20  0.80  ███░░░░░░ +0.42 ↑

        Args:
            contrib: IndicatorContribution with signal data.

        Returns:
            Markup string for the row.
        """
        # Signal name (truncate to 10 chars, left-aligned)
        name = contrib.name[:10].ljust(10)

        # Value (right-aligned, 6 chars)
        value_str = f"{contrib.value:6.2f}"

        # Weight (right-aligned, 4 chars)
        weight_str = f"{contrib.weight:4.2f}"

        # Contribution bar (9 chars)
        bar_length = 9
        fill_count = int(abs(contrib.contribution) * bar_length)
        fill_count = min(fill_count, bar_length)
        fill = "█" * fill_count
        empty = "░" * (bar_length - fill_count)

        # Determine color and direction
        if contrib.contribution > 0.01:
            bar = f"{empty}{fill}"
            color = "green"
            sign = "+"
            direction = "↑"
        elif contrib.contribution < -0.01:
            bar = f"{fill}{empty}"
            color = "red"
            sign = ""
            direction = "↓"
        else:
            bar = "░" * bar_length
            color = "dim"
            sign = ""
            direction = "→"

        # Contribution value (signed, 5 chars)
        contrib_str = f"{sign}{contrib.contribution:.2f}"

        # Build the row
        return (
            f"[dim]{name}[/dim] "
            f"{value_str} "
            f"{weight_str}  "
            f"[{color}]{bar} {contrib_str} {direction}[/{color}]"
        )
