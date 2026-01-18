"""
Position Detail Screen with strategy decision and risk context.

Combines position and strategy information:
- Current position details (entry, mark, P&L, leverage)
- Last strategy decision with indicators and confidence
- Risk metrics and active guards
- Recent trade history for this position
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import DataTable, Label, Static

from gpt_trader.tui.formatting import format_currency, format_price
from gpt_trader.tui.helpers import safe_update
from gpt_trader.tui.staleness_helpers import get_staleness_banner
from gpt_trader.tui.theme import THEME
from gpt_trader.tui.thresholds import (
    get_confidence_label,
    get_confidence_status,
    get_loss_ratio_status,
    get_status_color,
)
from gpt_trader.tui.utilities import (
    copy_to_clipboard,
    format_leverage_colored,
    format_pnl_colored,
)
from gpt_trader.tui.widgets import ContextualFooter
from gpt_trader.tui.widgets.shell import CommandBar
from gpt_trader.tui.widgets.tile_states import TileBanner, TileEmptyState
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState
    from gpt_trader.tui.types import DecisionData, Position, RiskState

logger = get_logger(__name__, component="tui")


class PositionCard(Static):
    """Card displaying current position details."""

    def compose(self) -> ComposeResult:
        """Compose the position card layout."""
        yield Label("ACTIVE POSITION", classes="widget-header")
        yield Label("", id="position-symbol", classes="position-symbol")

        with Grid(classes="position-metrics"):
            with Vertical(classes="metric"):
                yield Label("Side", classes="metric-label")
                yield Label("--", id="pos-side", classes="metric-value")

            with Vertical(classes="metric"):
                yield Label("Quantity", classes="metric-label")
                yield Label("--", id="pos-qty", classes="metric-value")  # naming: allow

            with Vertical(classes="metric"):
                yield Label("Entry Price", classes="metric-label")
                yield Label("--", id="pos-entry", classes="metric-value")

            with Vertical(classes="metric"):
                yield Label("Mark Price", classes="metric-label")
                yield Label("--", id="pos-mark", classes="metric-value")

            with Vertical(classes="metric"):
                yield Label("Unrealized P&L", classes="metric-label")
                yield Label("--", id="pos-pnl", classes="metric-value")

            with Vertical(classes="metric"):
                yield Label("Leverage", classes="metric-label")
                yield Label("--", id="pos-leverage", classes="metric-value")

        # For futures positions
        with Horizontal(classes="futures-info hidden", id="futures-info"):
            yield Label("Liquidation:", classes="info-label")
            yield Label("--", id="pos-liquidation", classes="info-value")
            yield Label("Buffer:", classes="info-label")
            yield Label("--", id="pos-buffer", classes="info-value")

    def update_position(self, position: Position | None) -> None:
        """Update position display."""
        if position is None:
            self.query_one("#position-symbol", Label).update("No Active Position")
            for label_id in [
                "pos-side",
                "pos-qty",  # naming: allow
                "pos-entry",
                "pos-mark",
                "pos-pnl",
                "pos-leverage",
            ]:
                self.query_one(f"#{label_id}", Label).update("--")
            return

        try:
            self.query_one("#position-symbol", Label).update(position.symbol)

            # Side with color
            side_color = (
                THEME.colors.success if position.side.lower() == "long" else THEME.colors.error
            )
            self.query_one("#pos-side", Label).update(
                Text.from_markup(f"[{side_color}]{position.side.upper()}[/{side_color}]")
            )

            # Quantity  # naming: allow
            qty = float(position.quantity)  # naming: allow
            qty_str = f"{qty:,.8f}" if qty < 1 else f"{qty:,.4f}"  # naming: allow
            self.query_one("#pos-qty", Label).update(qty_str)  # naming: allow

            # Prices
            self.query_one("#pos-entry", Label).update(format_price(position.entry_price))
            self.query_one("#pos-mark", Label).update(format_price(position.mark_price))

            # P&L with color
            pnl_text = format_pnl_colored(
                float(position.unrealized_pnl), format_currency(position.unrealized_pnl)
            )
            self.query_one("#pos-pnl", Label).update(pnl_text)

            # Leverage with color coding
            leverage_text = format_leverage_colored(position.leverage)
            self.query_one("#pos-leverage", Label).update(leverage_text)

            # Futures-specific info
            futures_info = self.query_one("#futures-info", Horizontal)
            if position.is_futures and position.liquidation_price:
                futures_info.remove_class("hidden")
                self.query_one("#pos-liquidation", Label).update(
                    format_price(position.liquidation_price)
                )
                buffer_pct = position.liquidation_buffer_pct or 0
                self.query_one("#pos-buffer", Label).update(f"{buffer_pct:.1f}%")
            else:
                futures_info.add_class("hidden")

        except Exception as e:
            logger.debug(f"Failed to update position card: {e}")


class StrategyDecisionCard(Static):
    """Card displaying last strategy decision with indicators."""

    # Indicator categories for semantic grouping
    INDICATOR_CATEGORIES = {
        "trend": ["trend", "crossover_signal", "short_ma", "long_ma", "adx"],
        "momentum": ["rsi", "rsi_signal", "momentum"],
        "regime": ["regime"],
        "order_flow": ["aggressor_ratio", "trade_count", "volume", "vwap"],
        "microstructure": ["spread_bps", "spread", "quality"],
    }

    def compose(self) -> ComposeResult:
        """Compose the strategy decision card layout."""
        yield Label("LAST STRATEGY DECISION", classes="widget-header")

        # Regime badge (shown when available, e.g., for ensemble strategies)
        yield Label("", id="decision-regime", classes="decision-regime hidden")

        with Horizontal(classes="decision-header"):
            yield Label("", id="decision-action", classes="decision-action")
            yield Label("", id="decision-confidence", classes="decision-confidence")

        yield Label("", id="decision-reason", classes="decision-reason")
        yield Label("", id="decision-time", classes="decision-time dim")

        yield Label("TOP MOVERS", classes="subsection-header", id="movers-header")
        yield Label("", id="decision-movers", classes="decision-movers")

        yield Label("INDICATORS", classes="subsection-header")
        yield DataTable(id="indicators-table", zebra_stripes=True)

    def on_mount(self) -> None:
        """Set up indicators table."""
        table = self.query_one("#indicators-table", DataTable)
        table.add_column("Indicator", key="indicator")
        table.add_column("Value", key="value")

    def update_decision(self, decision: DecisionData | None, symbol: str = "") -> None:
        """Update decision display."""
        regime_label = self.query_one("#decision-regime", Label)

        if decision is None or decision.symbol != symbol:
            self.query_one("#decision-action", Label).update("No recent decision")
            self.query_one("#decision-confidence", Label).update("")
            self.query_one("#decision-reason", Label).update("")
            self.query_one("#decision-time", Label).update("")
            self.query_one("#decision-movers", Label).update("")
            self.query_one("#movers-header", Label).add_class("hidden")
            regime_label.update("")
            regime_label.add_class("hidden")
            try:
                self.query_one("#indicators-table", DataTable).clear()
            except Exception:
                pass
            return

        try:
            # Regime badge (for ensemble strategies)
            regime = decision.indicators.get("regime")
            if regime:
                regime_icon = (
                    "📈" if regime == "trending" else "📊" if regime == "ranging" else "⚪"
                )
                regime_text = f"{regime_icon} {regime.upper()}"
                adx = decision.indicators.get("adx")
                if adx is not None:
                    regime_text += f" (ADX: {adx:.1f})"
                regime_label.update(regime_text)
                regime_label.remove_class("hidden")
            else:
                regime_label.update("")
                regime_label.add_class("hidden")

            # Action with color
            action = decision.action.upper()
            if action in ("BUY", "LONG"):
                action_markup = f"[{THEME.colors.success}]{action}[/{THEME.colors.success}]"
            elif action in ("SELL", "SHORT"):
                action_markup = f"[{THEME.colors.error}]{action}[/{THEME.colors.error}]"
            else:
                action_markup = f"[{THEME.colors.warning}]{action}[/{THEME.colors.warning}]"

            self.query_one("#decision-action", Label).update(Text.from_markup(action_markup))

            # Confidence with progress bar and badge
            confidence = decision.confidence
            conf_bar = self._make_confidence_bar(confidence)
            self.query_one("#decision-confidence", Label).update(conf_bar)

            # Reason
            self.query_one("#decision-reason", Label).update(
                decision.reason or "No reason provided"
            )

            # Timestamp
            if decision.timestamp > 0:
                dt = datetime.fromtimestamp(decision.timestamp)
                self.query_one("#decision-time", Label).update(f"at {dt.strftime('%H:%M:%S')}")
            else:
                self.query_one("#decision-time", Label).update("")

            # Top movers display
            movers_label = self.query_one("#decision-movers", Label)
            movers_header = self.query_one("#movers-header", Label)
            if decision.contributions:
                top_movers = decision.top_contributors
                if top_movers:
                    movers_header.remove_class("hidden")
                    mover_lines = []
                    for contrib in top_movers:
                        # Create visual contribution bar
                        bar = self._make_contribution_bar(contrib.contribution)
                        value_str = (
                            f"{contrib.value}"
                            if isinstance(contrib.value, str)
                            else f"{contrib.value:.2f}"
                        )
                        mover_lines.append(f"{contrib.name}: {bar} ({value_str})")
                    movers_label.update("\n".join(mover_lines))
                else:
                    movers_header.add_class("hidden")
                    movers_label.update("")
            else:
                movers_header.add_class("hidden")
                movers_label.update("")

            # Indicators table with semantic grouping
            table = self.query_one("#indicators-table", DataTable)
            table.clear()

            # Group indicators by category
            grouped = self._group_indicators(decision.indicators)
            for category, indicators in grouped.items():
                if not indicators:
                    continue
                # Add category header (skip regime as it's shown in badge)
                if category != "regime":
                    for name, value in indicators:
                        formatted_value = self._format_indicator_value(value)
                        table.add_row(name, formatted_value)

        except Exception as e:
            logger.debug(f"Failed to update decision card: {e}")

    def _make_confidence_bar(self, confidence: float) -> Text:
        """Create visual confidence bar with badge using shared thresholds."""
        # 0-1 scale
        if confidence > 1:
            confidence = confidence / 100  # Assume percentage

        filled = int(confidence * 10)
        bar = "●" * filled + "○" * (10 - filled)

        # Use shared threshold functions
        status = get_confidence_status(confidence)
        label = get_confidence_label(status)
        color = get_status_color(status)

        return Text.from_markup(
            f"[{color}]{bar}[/{color}] {confidence:.0%} [{color}]{label}[/{color}]"
        )

    def _make_contribution_bar(self, contribution: float) -> str:
        """Create visual contribution bar.

        Shows a bi-directional bar where:
        - Green bars on right = bullish contribution
        - Red bars on left = bearish contribution

        Args:
            contribution: Contribution score from -1.0 to +1.0

        Returns:
            Markup string with colored bar.
        """
        # Clamp to -1 to +1
        contribution = max(-1.0, min(1.0, contribution))

        # 5-segment bar (each direction)
        magnitude = int(abs(contribution) * 5)

        if contribution > 0:
            # Bullish (green, right side)
            bar = "     " + "[green]" + "▶" * magnitude + "[/green]"
            sign = "+"
        elif contribution < 0:
            # Bearish (red, left side)
            bar = "[red]" + "◀" * magnitude + "[/red]" + "     "[magnitude:]
            sign = ""
        else:
            bar = "  ○  "
            sign = ""

        return f"{bar} {sign}{contribution:.2f}"

    def _group_indicators(self, indicators: dict[str, Any]) -> dict[str, list[tuple[str, Any]]]:
        """Group indicators by semantic category.

        Args:
            indicators: Raw indicator dictionary from decision.

        Returns:
            Dictionary mapping category name to list of (indicator_name, value) tuples.
        """
        grouped: dict[str, list[tuple[str, Any]]] = {cat: [] for cat in self.INDICATOR_CATEGORIES}
        grouped["other"] = []

        for name, value in indicators.items():
            categorized = False
            for category, indicator_names in self.INDICATOR_CATEGORIES.items():
                if name.lower() in indicator_names:
                    grouped[category].append((name, value))
                    categorized = True
                    break
            if not categorized:
                grouped["other"].append((name, value))

        return grouped

    def _format_indicator_value(self, value: Any) -> str:
        """Format indicator value for display."""
        if isinstance(value, bool):
            return "✓" if value else "✗"
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, Decimal):
            return f"{float(value):.4f}"
        return str(value)


class RiskMetricsCard(Static):
    """Card displaying risk management metrics."""

    def compose(self) -> ComposeResult:
        """Compose the risk metrics card layout."""
        yield Label("RISK METRICS", classes="widget-header")

        with Grid(classes="risk-metrics"):
            with Vertical(classes="metric"):
                yield Label("Daily Loss", classes="metric-label")
                yield Label("--", id="risk-daily-loss", classes="metric-value")

            with Vertical(classes="metric"):
                yield Label("Daily Limit", classes="metric-label")
                yield Label("--", id="risk-daily-limit", classes="metric-value")

            with Vertical(classes="metric"):
                yield Label("Max Leverage", classes="metric-label")
                yield Label("--", id="risk-max-leverage", classes="metric-value")

            with Vertical(classes="metric"):
                yield Label("Reduce Only", classes="metric-label")
                yield Label("--", id="risk-reduce-only", classes="metric-value")

        yield Label("ACTIVE GUARDS", classes="subsection-header")
        yield Label("", id="active-guards", classes="guards-list")

    def update_risk(self, risk_state: RiskState | None) -> None:
        """Update risk metrics display."""
        if risk_state is None:
            return

        try:
            # Daily loss with color based on proximity to limit
            # Uses shared thresholds with abs() to correctly handle negative losses
            loss_pct = risk_state.current_daily_loss_pct
            limit_pct = risk_state.daily_loss_limit_pct

            # Use shared threshold function (correctly uses abs())
            loss_status = get_loss_ratio_status(loss_pct, limit_pct)
            loss_color = get_status_color(loss_status)

            self.query_one("#risk-daily-loss", Label).update(
                Text.from_markup(f"[{loss_color}]{loss_pct:.2f}%[/{loss_color}]")
            )
            self.query_one("#risk-daily-limit", Label).update(f"{limit_pct:.2f}%")

            # Max leverage
            self.query_one("#risk-max-leverage", Label).update(f"{risk_state.max_leverage:.1f}x")

            # Reduce only mode - unified display: "ON (reason)" / "OFF"
            if risk_state.reduce_only_mode:
                reason = risk_state.reduce_only_reason or "Risk limit"
                # Truncate long reasons
                if len(reason) > 20:
                    reason = reason[:17] + "..."
                self.query_one("#risk-reduce-only", Label).update(
                    Text.from_markup(f"[{THEME.colors.error}]ON[/{THEME.colors.error}] ({reason})")
                )
            else:
                self.query_one("#risk-reduce-only", Label).update(
                    Text.from_markup(f"[{THEME.colors.success}]OFF[/{THEME.colors.success}]")
                )

            # Active guards
            guard_names = [guard.name for guard in risk_state.guards if guard.name]
            if guard_names:
                guards_str = ", ".join(guard_names)
                self.query_one("#active-guards", Label).update(
                    Text.from_markup(
                        f"[{THEME.colors.warning}]{guards_str}[/{THEME.colors.warning}]"
                    )
                )
            else:
                self.query_one("#active-guards", Label).update("None active")

        except Exception as e:
            logger.debug(f"Failed to update risk card: {e}")


class PositionDetailScreen(Screen):
    """Position detail screen combining position, strategy, and risk info.

    Features:
    - Current position details (entry, mark, P&L, leverage)
    - Last strategy decision with indicators and confidence
    - Risk metrics and active guards
    - Recent trade history for this position

    Keyboard:
    - ESC/Q: Close and return to main screen
    - H: Toggle trade history panel
    - C: Copy position details
    - R: Refresh data
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("q", "dismiss", "Close"),
        Binding("h", "toggle_history", "Trade History", show=True),
        Binding("c", "copy_position", "Copy", show=True),
        Binding("r", "refresh", "Refresh", show=False),
    ]

    CSS = """
    #position-detail-container {
        layout: grid;
        grid-size: 2 2;
        grid-columns: 1fr 1fr;
        grid-rows: 1fr 1fr;
        height: 1fr;
        padding: 1;
    }

    #position-card-panel {
        border: round $border-primary;
        padding: 1;
    }

    #strategy-card-panel {
        border: round $border-primary;
        padding: 1;
    }

    #risk-card-panel {
        border: round $border-primary;
        padding: 1;
    }

    #history-panel {
        border: round $border-primary;
        padding: 1;
    }

    .position-symbol {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    .position-metrics, .risk-metrics {
        layout: grid;
        grid-size: 3 2;
        grid-gutter: 1;
        height: auto;
    }

    .metric {
        padding: 0 1;
    }

    .metric-label {
        color: $text-muted;
    }

    .metric-value {
        text-style: bold;
    }

    .decision-header {
        height: auto;
        margin-bottom: 1;
    }

    .decision-action {
        text-style: bold;
        margin-right: 2;
    }

    .decision-reason {
        margin: 1 0;
        color: $text-secondary;
    }

    .subsection-header {
        margin-top: 1;
        color: $text-muted;
        text-style: bold;
    }

    .guards-list {
        margin-top: 0;
    }

    .futures-info {
        margin-top: 1;
        height: auto;
    }

    .futures-info.hidden {
        display: none;
    }

    .info-label {
        color: $text-muted;
        margin-right: 1;
    }

    .info-value {
        margin-right: 2;
    }
    """

    # State
    state: reactive[TuiState | None] = reactive(None)

    # Toggle state
    show_history: reactive[bool] = reactive(False)

    def __init__(self, **kwargs) -> None:
        """Initialize PositionDetailScreen."""
        super().__init__(**kwargs)
        self._current_position: Position | None = None

    def compose(self) -> ComposeResult:
        """Compose the position detail screen layout."""
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(),
            id="header-bar",
        )
        yield TileBanner(id="position-detail-banner", classes="tile-banner hidden")

        with Container(id="position-detail-container"):
            # Top left: Position card
            with Container(id="position-card-panel"):
                yield PositionCard(id="position-card")

            # Top right: Strategy decision
            with Container(id="strategy-card-panel"):
                yield StrategyDecisionCard(id="strategy-card")

            # Bottom left: Risk metrics
            with Container(id="risk-card-panel"):
                yield RiskMetricsCard(id="risk-card")

            # Bottom right: Trade history
            with Vertical(id="history-panel"):
                yield Label("RECENT TRADES", classes="widget-header")
                table = DataTable(id="history-table", zebra_stripes=True)
                table.can_focus = True
                yield table
                yield TileEmptyState(
                    title="No Recent Trades",
                    subtitle="Trade history will appear here",
                    icon="○",
                    actions=["[R] Refresh"],
                    id="history-empty",
                )

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Initialize screen on mount."""
        logger.debug("PositionDetailScreen mounted")

        # Set up history table
        table = self.query_one("#history-table", DataTable)
        table.add_column("Time", key="time")
        table.add_column("Side", key="side")
        table.add_column("Price", key="price")
        table.add_column("Size", key="size")
        table.add_column("Fee", key="fee")

        # Register for state updates
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

        # Load initial state
        if hasattr(self.app, "tui_state"):
            self.state = self.app.tui_state  # type: ignore[attr-defined]

    def on_unmount(self) -> None:
        """Clean up on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry."""
        # Update staleness banner
        try:
            banner = self.query_one("#position-detail-banner", TileBanner)
            staleness_result = get_staleness_banner(state)
            if staleness_result:
                banner.update_banner(staleness_result[0], severity=staleness_result[1])
            else:
                banner.update_banner("")
        except Exception:
            pass

        self.state = state

    def watch_state(self, state: TuiState | None) -> None:
        """React to state changes - update all displays."""
        if state is None:
            return

        self._update_displays(state)

    @safe_update(notify_user=True, error_tracker=True, severity="warning")
    def _update_displays(self, state: TuiState) -> None:
        """Update all display panels from state."""
        # Find active position (first non-zero position)
        position = None
        positions = state.position_data.positions
        for pos in positions.values():
            if pos.quantity != 0:
                position = pos
                break

        self._current_position = position

        # Update position card
        try:
            pos_card = self.query_one("#position-card", PositionCard)
            pos_card.update_position(position)
        except Exception as e:
            logger.debug(f"Failed to update position card: {e}")

        # Update strategy decision card
        try:
            strategy_card = self.query_one("#strategy-card", StrategyDecisionCard)
            symbol = position.symbol if position else ""
            decision = state.strategy_data.last_decisions.get(symbol)
            strategy_card.update_decision(decision, symbol)
        except Exception as e:
            logger.debug(f"Failed to update strategy card: {e}")

        # Update risk card
        try:
            risk_card = self.query_one("#risk-card", RiskMetricsCard)
            risk_card.update_risk(state.risk_data)
        except Exception as e:
            logger.debug(f"Failed to update risk card: {e}")

        # Update trade history
        self._update_trade_history(state, position)

    def _update_trade_history(self, state: TuiState, position: Position | None) -> None:
        """Update trade history table filtered by position symbol."""
        table = self.query_one("#history-table", DataTable)
        empty_state = self.query_one("#history-empty", TileEmptyState)

        trades = state.trade_data.trades
        symbol = position.symbol if position else None

        # Filter trades by symbol if we have a position
        if symbol:
            filtered_trades = [t for t in trades if t.symbol == symbol]
        else:
            filtered_trades = trades[:10]  # Show last 10 if no position

        if not filtered_trades:
            table.display = False
            empty_state.display = True
            # Dynamic subtitle based on symbol context
            subtitle = f"No trades for {symbol}" if symbol else "Trade history will appear here"
            empty_state.update_state(subtitle=subtitle)
            return

        table.display = True
        empty_state.display = False
        table.clear()

        for trade in filtered_trades[:10]:  # Show last 10
            # Format time
            if trade.timestamp:
                dt = datetime.fromtimestamp(trade.timestamp)
                time_str = dt.strftime("%H:%M:%S")
            else:
                time_str = "--"

            # Side with color
            side_color = THEME.colors.success if trade.side.upper() == "BUY" else THEME.colors.error
            side_str = f"[{side_color}]{trade.side.upper()}[/{side_color}]"

            # Price and size
            price_str = format_price(trade.price)
            size_str = f"{float(trade.size):,.4f}"

            # Fee
            fee_str = format_currency(trade.fee) if trade.fee else "--"

            table.add_row(time_str, side_str, price_str, size_str, fee_str)

    # === Actions ===

    def action_dismiss(self) -> None:
        """Close screen and return to main."""
        self.app.pop_screen()

    def action_toggle_history(self) -> None:
        """Toggle trade history visibility."""
        self.show_history = not self.show_history
        try:
            history_panel = self.query_one("#history-panel", Vertical)
            if self.show_history:
                history_panel.remove_class("hidden")
            else:
                history_panel.add_class("hidden")
        except Exception:
            pass
        self.notify(f"Trade history {'shown' if self.show_history else 'hidden'}", timeout=2)

    def action_copy_position(self) -> None:
        """Copy position details to clipboard."""
        if self._current_position is None:
            self.notify("No position to copy", severity="warning", timeout=2)
            return

        pos = self._current_position
        text = (
            f"Symbol: {pos.symbol}\n"
            f"Side: {pos.side}\n"
            f"Quantity: {float(pos.quantity)}\n"
            f"Entry: {format_price(pos.entry_price)}\n"
            f"Mark: {format_price(pos.mark_price)}\n"
            f"P&L: {format_currency(pos.unrealized_pnl)}\n"
            f"Leverage: {pos.leverage}x"
        )

        if copy_to_clipboard(text):
            self.notify("Position copied to clipboard", timeout=2)
        else:
            self.notify("Copy failed", severity="warning", timeout=2)

    def action_refresh(self) -> None:
        """Manually refresh data via app-level reconnect."""
        if hasattr(self.app, "action_reconnect_data"):
            self.app.action_reconnect_data()
