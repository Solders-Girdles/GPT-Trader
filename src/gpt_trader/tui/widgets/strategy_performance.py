"""Strategy performance widget for dashboard display.

Displays key strategy performance metrics including win rate, profit factor,
returns, drawdown, and trade statistics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.types import RegimeData, StrategyPerformance

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


class StrategyPerformanceWidget(Static):
    """Displays strategy performance metrics.

    Layout:
    ┌─ STRATEGY PERFORMANCE ─────────────┐
    │ Win Rate     65.2%    Trades  23   │
    │ Profit Fct   1.85     W/L    15/8  │
    │ Return      +12.4%    Sharpe  1.2  │
    │ Drawdown    -3.2%     [BULL ↑]     │
    └────────────────────────────────────┘
    """

    # Track previous performance for delta calculation
    _last_performance: StrategyPerformance | None = None

    DEFAULT_CSS = """
    StrategyPerformanceWidget {
        height: auto;
        padding: 0 1;
    }

    StrategyPerformanceWidget .perf-header {
        height: 1;
        margin-bottom: 1;
    }

    StrategyPerformanceWidget .widget-header {
        text-style: bold;
        color: $text;
    }

    StrategyPerformanceWidget .perf-grid {
        height: auto;
        layout: grid;
        grid-size: 4;
        grid-columns: 1fr 1fr 1fr 1fr;
        grid-gutter: 0 1;
    }

    StrategyPerformanceWidget .metric-cell {
        height: 1;
    }

    StrategyPerformanceWidget .metric-label {
        color: $text-muted;
    }

    StrategyPerformanceWidget .metric-value {
        text-style: bold;
    }

    StrategyPerformanceWidget .metric-value.positive {
        color: $success;
    }

    StrategyPerformanceWidget .metric-value.negative {
        color: $error;
    }

    StrategyPerformanceWidget .metric-value.neutral {
        color: $text;
    }

    StrategyPerformanceWidget .regime-badge {
        text-style: bold;
        padding: 0 1;
    }

    StrategyPerformanceWidget .regime-badge.bullish {
        color: $success;
    }

    StrategyPerformanceWidget .regime-badge.bearish {
        color: $error;
    }

    StrategyPerformanceWidget .regime-badge.sideways {
        color: $warning;
    }

    StrategyPerformanceWidget .regime-badge.crisis {
        color: $error;
        text-style: bold reverse;
    }

    StrategyPerformanceWidget .empty-state {
        color: $text-muted;
        text-align: center;
        padding: 1;
    }
    """

    # Reactive state property for automatic updates
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update performance automatically."""
        if state is None:
            return
        self.update_display(state.strategy_performance, state.regime_data)

    def compose(self) -> ComposeResult:
        with Horizontal(classes="perf-header"):
            yield Label("STRATEGY PERFORMANCE", classes="widget-header")

        with Vertical(classes="perf-grid"):
            # Row 1: Win Rate, Trades
            yield Label("Win Rate", classes="metric-label metric-cell", id="lbl-win-rate")
            yield Label("--", classes="metric-value metric-cell", id="val-win-rate")
            yield Label("Trades", classes="metric-label metric-cell", id="lbl-trades")
            yield Label("--", classes="metric-value metric-cell", id="val-trades")

            # Row 2: Profit Factor, W/L
            yield Label("Profit Fct", classes="metric-label metric-cell", id="lbl-pf")
            yield Label("--", classes="metric-value metric-cell", id="val-pf")
            yield Label("W/L", classes="metric-label metric-cell", id="lbl-wl")
            yield Label("--", classes="metric-value metric-cell", id="val-wl")

            # Row 3: Return, Sharpe
            yield Label("Return", classes="metric-label metric-cell", id="lbl-return")
            yield Label("--", classes="metric-value metric-cell", id="val-return")
            yield Label("Sharpe", classes="metric-label metric-cell", id="lbl-sharpe")
            yield Label("--", classes="metric-value metric-cell", id="val-sharpe")

            # Row 4: Drawdown, Regime
            yield Label("Drawdown", classes="metric-label metric-cell", id="lbl-dd")
            yield Label("--", classes="metric-value metric-cell", id="val-dd")
            yield Label("", classes="metric-label metric-cell", id="lbl-regime")
            yield Label("--", classes="regime-badge metric-cell", id="val-regime")

    def on_mount(self) -> None:
        """Register with state registry for state updates."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from state registry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Called by StateRegistry when state changes."""
        self.state = state

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

    @safe_update
    def update_display(
        self,
        performance: StrategyPerformance,
        regime: RegimeData,
    ) -> None:
        """Update the widget display with performance metrics.

        Args:
            performance: Strategy performance metrics.
            regime: Current market regime data.
        """
        # Get previous values for delta calculation
        prev = self._last_performance

        # Win Rate
        win_rate_label = self.query_one("#val-win-rate", Label)
        if performance.total_trades > 0:
            prev_win_rate = prev.win_rate_pct if prev else None
            delta = self._format_delta(performance.win_rate_pct, prev_win_rate, suffix="%")
            win_rate_label.update(f"{performance.win_rate_pct:.1f}%{delta}")
            win_rate_label.remove_class("positive", "negative", "neutral")
            if performance.win_rate >= 0.5:
                win_rate_label.add_class("positive")
            else:
                win_rate_label.add_class("negative")
        else:
            win_rate_label.update("--")
            win_rate_label.remove_class("positive", "negative", "neutral")

        # Trades
        trades_label = self.query_one("#val-trades", Label)
        prev_trades = prev.total_trades if prev else None
        delta = self._format_delta(performance.total_trades, prev_trades, precision=0)
        trades_label.update(f"{performance.total_trades}{delta}")

        # Profit Factor
        pf_label = self.query_one("#val-pf", Label)
        if performance.profit_factor > 0:
            prev_pf = prev.profit_factor if prev else None
            delta = self._format_delta(performance.profit_factor, prev_pf, precision=2)
            pf_label.update(f"{performance.profit_factor:.2f}{delta}")
            pf_label.remove_class("positive", "negative", "neutral")
            if performance.profit_factor >= 1.5:
                pf_label.add_class("positive")
            elif performance.profit_factor >= 1.0:
                pf_label.add_class("neutral")
            else:
                pf_label.add_class("negative")
        else:
            pf_label.update("--")
            pf_label.remove_class("positive", "negative", "neutral")

        # W/L
        wl_label = self.query_one("#val-wl", Label)
        wl_label.update(f"{performance.winning_trades}/{performance.losing_trades}")

        # Return
        return_label = self.query_one("#val-return", Label)
        if performance.total_return_pct != 0:
            sign = "+" if performance.total_return_pct > 0 else ""
            prev_return = prev.total_return_pct if prev else None
            delta = self._format_delta(performance.total_return_pct, prev_return, suffix="%")
            return_label.update(f"{sign}{performance.total_return_pct:.1f}%{delta}")
            return_label.remove_class("positive", "negative", "neutral")
            if performance.total_return_pct > 0:
                return_label.add_class("positive")
            elif performance.total_return_pct < 0:
                return_label.add_class("negative")
            else:
                return_label.add_class("neutral")
        else:
            return_label.update("--")
            return_label.remove_class("positive", "negative", "neutral")

        # Sharpe
        sharpe_label = self.query_one("#val-sharpe", Label)
        if performance.sharpe_ratio != 0:
            prev_sharpe = prev.sharpe_ratio if prev else None
            delta = self._format_delta(performance.sharpe_ratio, prev_sharpe, precision=2)
            sharpe_label.update(f"{performance.sharpe_ratio:.2f}{delta}")
            sharpe_label.remove_class("positive", "negative", "neutral")
            if performance.sharpe_ratio >= 1.5:
                sharpe_label.add_class("positive")
            elif performance.sharpe_ratio >= 0.5:
                sharpe_label.add_class("neutral")
            else:
                sharpe_label.add_class("negative")
        else:
            sharpe_label.update("--")
            sharpe_label.remove_class("positive", "negative", "neutral")

        # Drawdown
        dd_label = self.query_one("#val-dd", Label)
        if performance.max_drawdown_pct != 0:
            prev_dd = prev.max_drawdown_pct if prev else None
            delta = self._format_delta(performance.max_drawdown_pct, prev_dd, suffix="%")
            dd_label.update(f"{performance.max_drawdown_pct:.1f}%{delta}")
            dd_label.remove_class("positive", "negative", "neutral")
            if performance.max_drawdown_pct > -5:
                dd_label.add_class("neutral")
            elif performance.max_drawdown_pct > -10:
                dd_label.add_class("negative")
            else:
                dd_label.add_class("negative")
        else:
            dd_label.update("--")
            dd_label.remove_class("positive", "negative", "neutral")

        # Regime
        regime_label = self.query_one("#val-regime", Label)
        if regime.regime != "UNKNOWN":
            regime_display = f"[{regime.short_label} {regime.icon}]"
            regime_label.update(regime_display)
            regime_label.remove_class("bullish", "bearish", "sideways", "crisis")
            if regime.is_crisis:
                regime_label.add_class("crisis")
            elif regime.is_bullish:
                regime_label.add_class("bullish")
            elif regime.is_bearish:
                regime_label.add_class("bearish")
            else:
                regime_label.add_class("sideways")
        else:
            regime_label.update("--")
            regime_label.remove_class("bullish", "bearish", "sideways", "crisis")

        # Store current performance for next delta calculation
        self._last_performance = performance
