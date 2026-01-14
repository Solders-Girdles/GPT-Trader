"""
Trading Statistics Widget.

Displays trading performance metrics with time window toggle
and sample size indicators for informed decision-making.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button, Label, Static

from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.services.trading_stats_service import (
    TIME_WINDOWS,
    TradingStatsService,
    get_trading_stats_service,
)
from gpt_trader.tui.types import TradingStats
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class TradingStatsWidget(Static):
    """Widget displaying trading performance metrics.

    Shows win rate, P&L, and trade statistics with time window
    toggle and sample size indicators.

    Keybinds:
        w: Cycle time window (all/5m/15m/30m/1h)
        W: Reset to all session
    """

    BINDINGS = [
        Binding("w", "cycle_window", "Window", show=True),
        Binding("W", "reset_window", "All Session", show=False),
    ]

    # Reactive stats for display updates
    stats: reactive[TradingStats | None] = reactive(None)

    def __init__(self, compact: bool = False, *args, **kwargs):
        """Initialize the trading stats widget.

        Args:
            compact: If True, show condensed single-line view.
        """
        super().__init__(*args, **kwargs)
        self._compact = compact
        self._service: TradingStatsService = get_trading_stats_service()
        self._has_received_update = False

    def compose(self) -> ComposeResult:
        with Horizontal(classes="stats-header"):
            yield Label("TRADING STATS", classes="widget-header")
            yield Label("", id="stats-window", classes="window-label")
            yield Label("", id="stats-sample", classes="sample-label")

        if self._compact:
            # Single-line compact view
            with Horizontal(classes="stats-compact"):
                yield Label("--", id="stats-winrate", classes="stat-value")
                yield Label("|", classes="stat-separator")
                yield Label("--", id="stats-pnl", classes="stat-value")
                yield Label("|", classes="stat-separator")
                yield Label("--", id="stats-trades", classes="stat-value")
        else:
            # Full expanded view
            with Vertical(classes="stats-body"):
                # Win/Loss Row
                with Horizontal(classes="stats-row"):
                    with Static(classes="stat-metric"):
                        yield Label("Win Rate", classes="stat-label")
                        yield Label("--", id="stats-winrate", classes="stat-value")
                    with Static(classes="stat-metric"):
                        yield Label("Win/Loss", classes="stat-label")
                        yield Label("--", id="stats-winloss", classes="stat-value")
                    with Static(classes="stat-metric"):
                        yield Label("Profit Factor", classes="stat-label")
                        yield Label("--", id="stats-pf", classes="stat-value")

                # P&L Row
                with Horizontal(classes="stats-row"):
                    with Static(classes="stat-metric"):
                        yield Label("Total P&L", classes="stat-label")
                        yield Label("--", id="stats-pnl", classes="stat-value")
                    with Static(classes="stat-metric"):
                        yield Label("Avg Win", classes="stat-label")
                        yield Label("--", id="stats-avgwin", classes="stat-value")
                    with Static(classes="stat-metric"):
                        yield Label("Avg Loss", classes="stat-label")
                        yield Label("--", id="stats-avgloss", classes="stat-value")

                # Trades Row
                with Horizontal(classes="stats-row"):
                    with Static(classes="stat-metric"):
                        yield Label("Total Trades", classes="stat-label")
                        yield Label("--", id="stats-trades", classes="stat-value")
                    with Static(classes="stat-metric"):
                        yield Label("Avg Size", classes="stat-label")
                        yield Label("--", id="stats-avgsize", classes="stat-value")

                # Time window chips
                with Horizontal(classes="window-chips"):
                    for minutes, label in TIME_WINDOWS:
                        chip_id = f"window-{minutes}"
                        active_class = "active" if minutes == 0 else ""
                        yield Button(
                            label,
                            id=chip_id,
                            classes=f"window-chip {active_class}".strip(),
                        )

    def on_mount(self) -> None:
        """Register with StateRegistry for updates."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from StateRegistry."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast."""
        self._has_received_update = True
        self.stats = self._service.compute_from_state(state)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle time window chip clicks."""
        button_id = event.button.id or ""
        if button_id.startswith("window-"):
            try:
                minutes = int(button_id.replace("window-", ""))
                # Find index for this window
                for i, (mins, _) in enumerate(TIME_WINDOWS):
                    if mins == minutes:
                        self._service._current_window_index = i
                        break
                self._update_window_chips()
                # Trigger re-computation by re-fetching state
                if hasattr(self.app, "state_registry"):
                    state = self.app.state_registry.get_state()
                    if state:
                        self.stats = self._service.compute_from_state(state)
                self.notify(f"Window: {self._service.current_window[1]}", timeout=2)
            except ValueError:
                pass

    def action_cycle_window(self) -> None:
        """Cycle through time windows."""
        _, label = self._service.cycle_window()
        self._update_window_chips()
        # Trigger re-computation
        if hasattr(self.app, "state_registry"):
            state = self.app.state_registry.get_state()
            if state:
                self.stats = self._service.compute_from_state(state)
        self.notify(f"Window: {label}", timeout=2)

    def action_reset_window(self) -> None:
        """Reset to 'All Session' window."""
        _, label = self._service.reset_window()
        self._update_window_chips()
        # Trigger re-computation
        if hasattr(self.app, "state_registry"):
            state = self.app.state_registry.get_state()
            if state:
                self.stats = self._service.compute_from_state(state)
        self.notify(f"Window: {label}", timeout=2)

    def _update_window_chips(self) -> None:
        """Update window chip active states."""
        current_minutes = self._service.current_window[0]
        for minutes, _ in TIME_WINDOWS:
            chip_id = f"window-{minutes}"
            try:
                chip = self.query_one(f"#{chip_id}", Button)
                if minutes == current_minutes:
                    chip.add_class("active")
                else:
                    chip.remove_class("active")
            except Exception:
                pass

    def watch_stats(self, stats: TradingStats | None) -> None:
        """Update display when stats change."""
        if stats is None:
            return

        # Update window label
        try:
            window_label = self.query_one("#stats-window", Label)
            window_label.update(f"[{stats.window_label}]")
        except Exception:
            pass

        # Update sample size with visual indicator for insufficient data
        try:
            sample_label = self.query_one("#stats-sample", Label)
            if stats.has_sufficient_data:
                sample_label.update(f"({stats.sample_label})")
                sample_label.remove_class("insufficient")
            else:
                sample_label.update(f"[dim]({stats.sample_label})[/dim]")
                sample_label.add_class("insufficient")
        except Exception:
            pass

        if self._compact:
            self._update_compact_view(stats)
        else:
            self._update_expanded_view(stats)

    def _update_compact_view(self, stats: TradingStats) -> None:
        """Update compact single-line display."""
        try:
            # Win rate with sample context
            winrate = self.query_one("#stats-winrate", Label)
            if stats.total_trades > 0:
                wr_pct = stats.win_rate * 100
                wr_color = "green" if wr_pct >= 50 else "yellow" if wr_pct >= 40 else "red"
                winrate.update(f"[{wr_color}]{wr_pct:.0f}% WR[/]")
            else:
                winrate.update("[dim]-- WR[/dim]")

            # Total P&L
            pnl = self.query_one("#stats-pnl", Label)
            if stats.total_trades > 0:
                pnl_color = (
                    "green" if stats.total_pnl > 0 else "red" if stats.total_pnl < 0 else "dim"
                )
                pnl.update(f"[{pnl_color}]{format_currency(stats.total_pnl)}[/]")
            else:
                pnl.update("[dim]$0[/dim]")

            # Trade count
            trades = self.query_one("#stats-trades", Label)
            trades.update(f"{stats.total_trades} trades")
        except Exception as e:
            logger.debug("Failed to update compact stats: %s", e)

    def _update_expanded_view(self, stats: TradingStats) -> None:
        """Update full expanded display."""
        try:
            # Win rate
            winrate = self.query_one("#stats-winrate", Label)
            if stats.total_trades > 0:
                wr_pct = stats.win_rate * 100
                wr_color = "green" if wr_pct >= 50 else "yellow" if wr_pct >= 40 else "red"
                winrate.update(f"[{wr_color}]{wr_pct:.1f}%[/]")
            else:
                winrate.update("[dim]--[/dim]")

            # Win/Loss counts
            winloss = self.query_one("#stats-winloss", Label)
            winloss.update(f"{stats.winning_trades}W / {stats.losing_trades}L")

            # Profit factor
            pf = self.query_one("#stats-pf", Label)
            if stats.profit_factor == float("inf"):
                pf.update("[green]∞[/green]")
            elif stats.profit_factor > 0:
                pf_color = (
                    "green"
                    if stats.profit_factor >= 1.5
                    else "yellow"
                    if stats.profit_factor >= 1
                    else "red"
                )
                pf.update(f"[{pf_color}]{stats.profit_factor:.2f}[/]")
            else:
                pf.update("[dim]--[/dim]")

            # Total P&L
            pnl = self.query_one("#stats-pnl", Label)
            if stats.total_trades > 0:
                pnl_color = (
                    "green" if stats.total_pnl > 0 else "red" if stats.total_pnl < 0 else "dim"
                )
                pnl.update(f"[{pnl_color}]{format_currency(stats.total_pnl)}[/]")
            else:
                pnl.update("[dim]$0[/dim]")

            # Average win
            avgwin = self.query_one("#stats-avgwin", Label)
            if stats.avg_win > 0:
                avgwin.update(f"[green]+{format_currency(stats.avg_win)}[/green]")
            else:
                avgwin.update("[dim]--[/dim]")

            # Average loss
            avgloss = self.query_one("#stats-avgloss", Label)
            if stats.avg_loss > 0:
                avgloss.update(f"[red]-{format_currency(stats.avg_loss)}[/red]")
            else:
                avgloss.update("[dim]--[/dim]")

            # Trade count
            trades = self.query_one("#stats-trades", Label)
            trades.update(str(stats.total_trades))

            # Average trade size
            avgsize = self.query_one("#stats-avgsize", Label)
            if stats.avg_trade_size > 0:
                avgsize.update(f"{stats.avg_trade_size:.4f}")
            else:
                avgsize.update("[dim]--[/dim]")
        except Exception as e:
            logger.debug("Failed to update expanded stats: %s", e)
