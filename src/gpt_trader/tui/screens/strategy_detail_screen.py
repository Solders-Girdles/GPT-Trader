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
    StrategyParameters,
    StrategyPerformance,
)

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


# Static tuning hints keyed by indicator name (uppercase) - fallback when no live params
TUNING_HINTS: dict[str, str] = {
    "RSI": "Higher period = slower signals",
    "MACD": "Wider spread = smoother trend",
    "ADX": "Higher period = smoother regime",
    "VWAP": "Longer window = slower mean",
    "BOLL": "Wider bands = fewer signals",
    "EMA": "Longer EMA = slower trend",
    "MA": "Longer MA = slower trend",
    "ZSCORE": "Higher threshold = fewer signals",
    "MEAN": "Higher threshold = fewer signals",
    "SPREAD": "Tighter spread = stricter filter",
    "ORDERBOOK": "More levels = smoother signal",
    "TREND": "Wider spread = smoother signal",
}


def _get_indicator_hint(name: str, params: StrategyParameters | None = None) -> str | None:
    """Get tuning hint for an indicator, preferring live params.

    When live params are available, formats them for display
    (e.g., "period=14"). Falls back to static hints when no params.

    Args:
        name: Indicator name (may include parameters).
        params: Optional live StrategyParameters for real config values.

    Returns:
        Hint string with live params or static hint, None if neither available.
    """
    # Try live params first
    if params is not None:
        live_hint = params.format_indicator_params(name)
        if live_hint:
            return live_hint

    # Fall back to static hints
    import re

    # Extract first alphabetic token
    match = re.match(r"([A-Za-z]+)", name)
    if match:
        key = match.group(1).upper()
        return TUNING_HINTS.get(key)
    return None


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

    StrategyDetailScreen .signal-detail-hint-note {
        color: $text-muted;
        text-style: dim italic;
        height: 1;
        margin-bottom: 1;
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
        self._last_performance: StrategyPerformance | None = None
        self._strategy_params: StrategyParameters | None = None

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

            # Backtest section (placeholder until backend supplies data)
            yield Label("BACKTEST", classes="section-header")
            with Container(classes="perf-grid", id="backtest-grid"):
                yield Label("Win Rate:", classes="perf-label")
                yield Label("--", id="backtest-win-rate", classes="perf-value")
                yield Label("Trades:", classes="perf-label")
                yield Label("--", id="backtest-trades", classes="perf-value")

                yield Label("Profit Factor:", classes="perf-label")
                yield Label("--", id="backtest-profit-factor", classes="perf-value")
                yield Label("Drawdown:", classes="perf-label")
                yield Label("--", id="backtest-drawdown", classes="perf-value")

            yield Label("No backtest data available", id="backtest-note", classes="muted")

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
                yield Label(
                    "Hints: tuned defaults shown; no live config",
                    classes="signal-detail-hint-note",
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
        # Store strategy params for hint formatting
        self._strategy_params = state.strategy_data.parameters

        self._update_header(state)
        self._update_performance(state.strategy_performance)
        self._update_regime(state.regime_data)
        self._update_decisions(state)
        self._update_backtest(state)
        self._update_hint_note()

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

    def _format_delta(
        self,
        current: float,
        previous: float | None,
        precision: int = 1,
        suffix: str = "",
    ) -> str:
        """Format a delta indicator for metric changes.

        Args:
            current: Current value.
            previous: Previous value (None if no previous).
            precision: Decimal places for display.
            suffix: Optional suffix (e.g., "%" for percentages).

        Returns:
            Formatted delta string (e.g., " [green]+1.2%[/green]") or empty string.
        """
        if previous is None:
            return ""

        delta = current - previous
        if abs(delta) < 0.01:  # Ignore tiny changes
            return ""

        sign = "+" if delta > 0 else ""
        color = "green" if delta > 0 else "red"
        return f" [{color}]{sign}{delta:.{precision}f}{suffix}[/{color}]"

    def _get_entry_exit_badge(self, action: str) -> str:
        """Get entry/exit badge for an action.

        Args:
            action: The action string (BUY, SELL, CLOSE, EXIT, HOLD).

        Returns:
            Badge string like " [cyan]ENTRY[/cyan]" or " [magenta]EXIT[/magenta]", or empty.
        """
        action_upper = action.upper()
        if action_upper in ("BUY", "SELL"):
            return " [cyan]ENTRY[/cyan]"
        elif action_upper in ("CLOSE", "EXIT"):
            return " [magenta]EXIT[/magenta]"
        return ""

    @safe_update
    def _update_performance(self, performance: StrategyPerformance) -> None:
        """Update the performance section."""
        prev = self._last_performance

        # Win Rate
        win_rate = self.query_one("#perf-win-rate", Label)
        if performance.total_trades > 0:
            prev_win_rate = prev.win_rate_pct if prev else None
            delta = self._format_delta(performance.win_rate_pct, prev_win_rate, suffix="%")
            win_rate.update(f"{performance.win_rate_pct:.1f}%{delta}")
            win_rate.remove_class("positive", "negative")
            win_rate.add_class("positive" if performance.win_rate >= 0.5 else "negative")
        else:
            win_rate.update("--")
            win_rate.remove_class("positive", "negative")

        # Trades
        trades = self.query_one("#perf-trades", Label)
        prev_trades = prev.total_trades if prev else None
        delta = self._format_delta(performance.total_trades, prev_trades, precision=0)
        trades.update(f"{performance.total_trades}{delta}")

        # Return
        ret = self.query_one("#perf-return", Label)
        if performance.total_return_pct != 0:
            sign = "+" if performance.total_return_pct > 0 else ""
            prev_return = prev.total_return_pct if prev else None
            delta = self._format_delta(performance.total_return_pct, prev_return, suffix="%")
            ret.update(f"{sign}{performance.total_return_pct:.1f}%{delta}")
            ret.remove_class("positive", "negative")
            ret.add_class("positive" if performance.total_return_pct > 0 else "negative")
        else:
            ret.update("--")
            ret.remove_class("positive", "negative")

        # Sharpe
        sharpe = self.query_one("#perf-sharpe", Label)
        if performance.sharpe_ratio != 0:
            prev_sharpe = prev.sharpe_ratio if prev else None
            delta = self._format_delta(performance.sharpe_ratio, prev_sharpe, precision=2)
            sharpe.update(f"{performance.sharpe_ratio:.2f}{delta}")
        else:
            sharpe.update("--")

        # Drawdown
        dd = self.query_one("#perf-drawdown", Label)
        if performance.max_drawdown_pct != 0:
            prev_dd = prev.max_drawdown_pct if prev else None
            delta = self._format_delta(performance.max_drawdown_pct, prev_dd, suffix="%")
            dd.update(f"{performance.max_drawdown_pct:.1f}%{delta}")
            dd.remove_class("positive", "negative")
            dd.add_class("negative")
        else:
            dd.update("--")
            dd.remove_class("positive", "negative")

        # W/L
        wl = self.query_one("#perf-wl", Label)
        wl.update(f"{performance.winning_trades}/{performance.losing_trades}")

        # Store current performance for next delta calculation
        self._last_performance = performance

    @safe_update
    def _update_regime(self, regime: RegimeData) -> None:
        """Update the regime badge."""
        badge = self.query_one("#regime-badge", Label)

        if regime.regime != "UNKNOWN":
            badge.update(f"[{regime.short_label} {regime.icon}] {regime.confidence * 100:.0f}%")
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
                color = "green"
            elif action == "SELL":
                color = "red"
            elif action in ("CLOSE", "EXIT"):
                color = "yellow"
            else:
                color = "yellow"

            # Entry/Exit badge
            badge = self._get_entry_exit_badge(action)
            formatted_action = f"[{color}]{action}[/{color}]{badge}"

            # Format confidence
            confidence = f"{decision.confidence:.2f}"

            # Format timestamp
            time_str = ""
            if decision.timestamp > 0:
                time_str = datetime.fromtimestamp(decision.timestamp).strftime("%H:%M:%S")

            table.add_row(
                symbol, formatted_action, confidence, decision.reason, time_str, key=symbol
            )

    def _build_backtest_display(
        self, performance: StrategyPerformance | None
    ) -> tuple[dict[str, str], str]:
        """Build backtest display values and note.

        Args:
            performance: StrategyPerformance with backtest data, or None.

        Returns:
            Tuple of (values dict, note string).
            Values dict keys: win_rate, trades, profit_factor, drawdown.
        """
        if performance is None or performance.total_trades == 0:
            return {
                "win_rate": "--",
                "trades": "--",
                "profit_factor": "--",
                "drawdown": "--",
            }, "No backtest data available"

        return {
            "win_rate": f"{performance.win_rate_pct:.1f}%",
            "trades": str(performance.total_trades),
            "profit_factor": f"{performance.profit_factor:.2f}",
            "drawdown": f"{performance.max_drawdown_pct:.1f}%",
        }, ""

    @safe_update
    def _update_backtest(self, state: TuiState) -> None:
        """Update the backtest section."""
        # Get backtest performance if available (placeholder for future backend data)
        backtest = getattr(state, "backtest_performance", None)
        values, note = self._build_backtest_display(backtest)

        # Update labels
        self.query_one("#backtest-win-rate", Label).update(values["win_rate"])
        self.query_one("#backtest-trades", Label).update(values["trades"])
        self.query_one("#backtest-profit-factor", Label).update(values["profit_factor"])
        self.query_one("#backtest-drawdown", Label).update(values["drawdown"])

        # Update note visibility
        note_label = self.query_one("#backtest-note", Label)
        if note:
            note_label.update(note)
            note_label.remove_class("hidden")
        else:
            note_label.add_class("hidden")

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
                RSI       35.20  0.80  ███░░░░░░ +0.42 ↑  (hint)

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
        row = (
            f"[dim]{name}[/dim] "
            f"{value_str} "
            f"{weight_str}  "
            f"[{color}]{bar} {contrib_str} {direction}[/{color}]"
        )

        # Append tuning hint if available (live params preferred, static fallback)
        hint = _get_indicator_hint(contrib.name, self._strategy_params)
        if hint:
            row += f"  [dim]({hint})[/dim]"

        return row

    @safe_update
    def _update_hint_note(self) -> None:
        """Update the hint note to reflect live vs static config status."""
        try:
            note = self.query_one(".signal-detail-hint-note", Label)
            if self._strategy_params is not None:
                note.update("Hints: live config parameters shown")
            else:
                note.update("Hints: tuned defaults shown; no live config")
        except Exception:
            pass
