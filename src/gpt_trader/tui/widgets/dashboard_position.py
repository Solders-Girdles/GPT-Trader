"""
Position-related dashboard widgets.

Contains:
- PositionCardWidget: Hero widget displaying the active position in detail
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.staleness_helpers import (
    get_connection_banner,
    get_empty_state_config,
    get_freshness_display,
    get_staleness_banner,
)
from gpt_trader.tui.widgets.tile_states import TileBanner, tile_empty_state, tile_loading_state
from gpt_trader.tui.widgets.value_flash import flash_label
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


class PositionCardWidget(Static):
    """
    Hero widget displaying the active position in detail.
    Designed for the top-left 'Hero' tile.

    Implements StateObserver to receive updates via StateRegistry broadcast.

    Styles are defined in styles/widgets/dashboard.tcss for centralized theming.
    """

    SCOPED_CSS = False  # Use global styles from dashboard.tcss

    position_data = reactive(None, always_update=True)  # type: ignore[var-annotated]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._has_received_update = False
        self._bot_running = False
        self._data_source_mode = "demo"
        self._connection_status = "UNKNOWN"
        # Track previous PnL for flash animation
        self._prev_pnl: float | None = None
        self._last_pnl_flash: float = 0.0  # Rate limit PnL flashing
        # Track last update for freshness display
        self._last_position_update: float = 0.0
        # Display signature cache for early-exit optimization
        self._last_display_signature: tuple | None = None

    def on_mount(self) -> None:
        """Register with StateRegistry on mount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from StateRegistry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def _update_position_freshness(self, state: TuiState) -> None:
        """Update position data freshness indicator in header."""
        try:
            indicator = self.query_one("#position-data-state", Label)
            freshness = get_freshness_display(state)

            if freshness:
                text, css_class = freshness
                indicator.update(text)
                indicator.remove_class("fresh", "stale", "critical")
                indicator.add_class(css_class)
            else:
                indicator.update("")
                indicator.remove_class("fresh", "stale", "critical")
        except Exception:
            pass

    def _compute_display_signature(self, state: TuiState) -> tuple:
        """Compute a signature from all fields displayed by this widget.

        Returns a tuple that can be compared for equality to detect changes.
        """
        # Get first position signature if available
        pos_sig = ()
        if state.position_data and state.position_data.positions:
            first_symbol = next(iter(state.position_data.positions))
            pos = state.position_data.positions[first_symbol]
            pos_sig = (
                pos.symbol,
                pos.side,
                pos.leverage,
                float(pos.unrealized_pnl),
                float(pos.entry_price),
                float(pos.mark_price),
                float(pos.liquidation_price) if pos.liquidation_price else 0.0,
            )

        # Connection status for banner logic
        conn_status = ""
        try:
            conn_status = str(getattr(state.system_data, "connection_status", "") or "")
        except Exception:
            pass

        return (
            state.running,
            state.data_source_mode,
            conn_status,
            state.degraded_mode,
            state.last_update_timestamp,
            pos_sig,
        )

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts position data from TuiState.position_data (PortfolioSummary)
        and converts to the dict format expected by watch_position_data().
        """
        # Early exit if display signature unchanged
        sig = self._compute_display_signature(state)
        if sig == self._last_display_signature:
            return
        self._last_display_signature = sig

        self._has_received_update = True
        self._bot_running = bool(getattr(state, "running", False))
        self._data_source_mode = str(getattr(state, "data_source_mode", "demo") or "demo")
        try:
            self._connection_status = str(getattr(state.system_data, "connection_status", "") or "")
        except Exception:
            self._connection_status = ""

        # Banner reflects overall connection health for this tile.
        try:
            banner = self.query_one(TileBanner)

            # Use shared helpers for consistent banner logic
            staleness_result = get_staleness_banner(state)
            if staleness_result:
                banner.update_banner(staleness_result[0], severity=staleness_result[1])
            else:
                conn_result = get_connection_banner(
                    self._connection_status,
                    self._bot_running,
                    state.degraded_mode,
                )
                if conn_result:
                    banner.update_banner(conn_result[0], severity=conn_result[1])
                else:
                    banner.update_banner("")
        except Exception as e:
            logger.debug("PositionCardWidget banner update failed: %s", e, exc_info=True)

        # Update data freshness indicator
        self._update_position_freshness(state)

        if not state.position_data.positions:
            self.position_data = None
            return

        # Get the first position (hero display shows one position)
        first_symbol = next(iter(state.position_data.positions))
        pos = state.position_data.positions[first_symbol]

        pos_dict = {
            "symbol": pos.symbol,
            "side": pos.side.upper() if pos.side else "LONG",
            "leverage": pos.leverage,
            "pnl": float(pos.unrealized_pnl),
            "entry_price": float(pos.entry_price),
            "mark_price": float(pos.mark_price),
            "liquidation_price": float(pos.liquidation_price) if pos.liquidation_price else 0.0,
        }
        self.position_data = pos_dict

    def compose(self) -> ComposeResult:
        with Horizontal(classes="position-header"):
            yield Label("POSITION", classes="widget-header")
            yield Label("", id="position-data-state", classes="data-state-label")
        yield TileBanner(id="position-banner", classes="tile-banner hidden")
        with Vertical(id="pos-body"):
            yield tile_loading_state("Waiting for position data…")

    def watch_position_data(self, pos: dict | None) -> None:
        try:
            body = self.query_one("#pos-body", Vertical)
        except Exception:
            return

        try:
            body.remove_children()

            if not self._has_received_update:
                body.mount(tile_loading_state("Waiting for position data…"))
                return

            if not pos:
                # Use shared empty state config for stopped/failed states
                if not self._bot_running:
                    config = get_empty_state_config(
                        data_type="Position",
                        bot_running=self._bot_running,
                        data_source_mode=self._data_source_mode,
                        connection_status=self._connection_status,
                    )
                    body.mount(
                        tile_empty_state(
                            config["title"],
                            config["subtitle"],
                            icon=config["icon"],
                            actions=config["actions"],
                        )
                    )
                    return

                conn = self._connection_status.upper()
                if conn in ("DISCONNECTED", "ERROR", "FAILED"):
                    body.mount(
                        tile_empty_state(
                            "Connection Failed",
                            "Check credentials and network",
                            icon="⚠",
                            actions=["[R] Reconnect", "[C] Config"],
                        )
                    )
                    return

                # Bot is running but no position - show Strategy + Portfolio snapshots
                body.mount(self._build_no_position_dashboard())
                return

            body.mount(self._build_active_state(pos))

            # Flash PnL if it changed (rate limited to 1 per second)
            import time as time_mod

            current_pnl = float(pos.get("pnl", 0.0))
            now = time_mod.time()
            if self._prev_pnl is not None and current_pnl != self._prev_pnl:
                if (now - self._last_pnl_flash) > 1.0:
                    try:
                        pnl_label = self.query_one("#pnl-hero-label", Label)
                        direction = "up" if current_pnl > self._prev_pnl else "down"
                        flash_label(pnl_label, direction=direction, duration=0.5)
                        self._last_pnl_flash = now
                    except Exception:
                        pass  # Label may not be mounted yet
            self._prev_pnl = current_pnl
        except Exception as e:
            logger.warning("PositionCardWidget render failed: %s", e, exc_info=True)
            try:
                body.remove_children()
                body.mount(
                    tile_empty_state(
                        "Position Tile Error",
                        "Render failed — see logs",
                        icon="⚠",
                        actions=["[R] Reconnect", "[L] Logs"],
                    )
                )
                try:
                    self.query_one(TileBanner).update_banner(
                        "Position tile render error",
                        severity="error",
                    )
                except Exception:
                    pass
            except Exception:
                pass

    def _build_empty_state(self) -> Vertical:
        """Build empty state widget with proper Textual compose pattern."""
        return tile_empty_state(
            "No Active Position",
            "Start the bot to begin trading",
            icon="◇",
            actions=["[S] Start Bot", "[C] Config"],
        )

    def _build_quick_actions(self) -> Vertical:
        """Build quick actions container when stopped and no position."""
        actions = Vertical(classes="quick-actions")
        actions.compose_add_child(Label("Quick Actions", classes="quick-actions-title"))

        row = Horizontal(classes="quick-actions-row")
        row.compose_add_child(Label("[S] Start Bot", classes="action-hint"))
        row.compose_add_child(Label("[R] Refresh Data", classes="action-hint"))
        row.compose_add_child(Label("[C] Config", classes="action-hint"))
        actions.compose_add_child(row)

        return actions

    def _build_no_position_dashboard(self) -> Vertical:
        """Build dashboard when bot is running but no active position.

        Shows Strategy Snapshot (last signals) and Portfolio Snapshot (cash, holdings).
        """
        root = Vertical(classes="no-position-dashboard")

        # Title row
        root.compose_add_child(Label("No Active Position", classes="no-pos-title"))
        root.compose_add_child(
            Label("System monitoring market conditions", classes="no-pos-subtitle")
        )

        # Strategy Snapshot section
        strategy_section = Vertical(classes="snapshot-section")
        strategy_section.compose_add_child(Label("STRATEGY", classes="snapshot-header"))

        if hasattr(self.app, "tui_state"):
            decisions = self.app.tui_state.strategy_data.last_decisions
            if decisions:
                for symbol, dec in list(decisions.items())[:2]:
                    color = (
                        "green"
                        if dec.action == "BUY"
                        else "red" if dec.action == "SELL" else "yellow"
                    )
                    confidence_str = f"{dec.confidence:.0%}" if dec.confidence else "--"
                    strategy_section.compose_add_child(
                        Label(
                            f"{symbol}: [{color}]{dec.action}[/{color}] ({confidence_str})",
                            classes="snapshot-row",
                        )
                    )
            else:
                strategy_section.compose_add_child(
                    Label("Awaiting signals...", classes="snapshot-row muted")
                )
        else:
            strategy_section.compose_add_child(
                Label("Awaiting signals...", classes="snapshot-row muted")
            )

        root.compose_add_child(strategy_section)

        # Portfolio Snapshot section
        portfolio_section = Vertical(classes="snapshot-section")
        portfolio_section.compose_add_child(Label("PORTFOLIO", classes="snapshot-header"))

        if hasattr(self.app, "tui_state"):
            state = self.app.tui_state

            # Cash balance (USD or USDC)
            usd_balance = next(
                (b for b in state.account_data.balances if b.asset in ("USD", "USDC")),
                None,
            )
            if usd_balance:
                portfolio_section.compose_add_child(
                    Label(f"Cash: {format_currency(usd_balance.available)}", classes="snapshot-row")
                )

            # Equity
            if state.position_data.equity:
                portfolio_section.compose_add_child(
                    Label(
                        f"Equity: {format_currency(state.position_data.equity)}",
                        classes="snapshot-row",
                    )
                )

            # Top crypto holdings (excluding USD/USDC)
            holdings = [
                b
                for b in state.account_data.balances
                if b.asset not in ("USD", "USDC") and float(b.total) > 0
            ][:3]
            if holdings:
                holding_names = ", ".join(b.asset for b in holdings)
                portfolio_section.compose_add_child(
                    Label(f"Holdings: {holding_names}", classes="snapshot-row")
                )
            elif not usd_balance and not state.position_data.equity:
                portfolio_section.compose_add_child(
                    Label("No data available", classes="snapshot-row muted")
                )
        else:
            portfolio_section.compose_add_child(
                Label("No data available", classes="snapshot-row muted")
            )

        root.compose_add_child(portfolio_section)

        return root

    def _build_active_state(self, pos: dict) -> Vertical:
        """Build active position state widget with proper Textual compose pattern."""
        symbol = pos.get("symbol", "BTC-USD")
        side = pos.get("side", "LONG").upper()
        leverage = pos.get("leverage", 1)
        pnl = float(pos.get("pnl", 0.0))

        entry = float(pos.get("entry_price", 0))
        mark = float(pos.get("mark_price", 0))
        liq = float(pos.get("liquidation_price", 0))

        side_cls = "long" if side == "LONG" else "short"
        pnl_fmt = format_currency(pnl)
        color = "green" if pnl >= 0 else "red"
        liq_fmt = format_currency(liq) if liq > 0 else "-"

        # Build container hierarchy properly
        root = Vertical()

        # 1. Header Row
        header_row = Horizontal(classes="pos-header-row")
        header_row.compose_add_child(
            Label(f" {side} {leverage}x ", classes=f"pos-badge {side_cls}")
        )
        header_row.compose_add_child(Label(f" {symbol} ", classes="pos-symbol"))
        root.compose_add_child(header_row)

        # 2. PnL Hero (with ID for flash animation)
        root.compose_add_child(
            Label(f"[{color}]💰 {pnl_fmt}[/]", id="pnl-hero-label", classes="pnl-hero")
        )

        # 3. Details Grid
        details_grid = Horizontal(classes="pos-details-grid")

        # Entry item
        entry_item = Vertical(classes="detail-item")
        entry_item.compose_add_child(Label("Entry", classes="detail-label"))
        entry_item.compose_add_child(Label(format_currency(entry), classes="detail-value"))
        details_grid.compose_add_child(entry_item)

        # Mark item
        mark_item = Vertical(classes="detail-item")
        mark_item.compose_add_child(Label("Mark", classes="detail-label"))
        mark_item.compose_add_child(Label(format_currency(mark), classes="detail-value"))
        details_grid.compose_add_child(mark_item)

        # Liq item
        liq_item = Vertical(classes="detail-item")
        liq_item.compose_add_child(Label("Liq", classes="detail-label"))
        liq_item.compose_add_child(Label(liq_fmt, classes="detail-value"))
        details_grid.compose_add_child(liq_item)

        root.compose_add_child(details_grid)
        return root
