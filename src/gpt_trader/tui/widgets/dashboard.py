"""
Dashboard widgets for the High-Fidelity TUI Bento Layout.

All widgets implement StateObserver protocol to receive state updates
via StateRegistry broadcast instead of direct property assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.staleness_helpers import (
    get_connection_banner,
    get_empty_state_config,
    get_freshness_display,
    get_staleness_banner,
)
from gpt_trader.tui.thresholds import DEFAULT_THRESHOLDS as PERF_THRESHOLDS
from gpt_trader.tui.thresholds import (
    get_error_rate_status,
    get_latency_status,
    get_status_color,
    get_status_icon,
)
from gpt_trader.tui.widgets.primitives import ProgressBarWidget, SparklineWidget
from gpt_trader.tui.widgets.tile_states import TileBanner, tile_empty_state, tile_loading_state
from gpt_trader.tui.widgets.value_flash import flash_label
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


def calculate_price_change_percent(
    current_price: float,
    history: list[Decimal] | list[float],
) -> float:
    """Calculate percentage change from oldest price in history to current.

    Since the price history is a rolling window (not timestamped 24h data),
    this calculates change from the oldest available price to current.

    Args:
        current_price: Current price value.
        history: List of historical prices (oldest first).

    Returns:
        Percentage change (e.g., 2.5 for +2.5%, -1.2 for -1.2%).
        Returns 0.0 if history is empty or oldest price is zero.
    """
    if not history:
        return 0.0

    # Get oldest price from history
    try:
        oldest_price = float(history[0])
    except (ValueError, TypeError, IndexError):
        return 0.0

    if oldest_price == 0.0:
        return 0.0

    # Calculate percentage change: ((new - old) / old) * 100
    return ((current_price - oldest_price) / oldest_price) * 100


if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState


class TickerRow(Static):
    """A single row representing a market ticker.

    Supports in-place updates via the update_values() method to avoid
    full widget recreation on data changes.
    """

    def __init__(
        self,
        symbol: str,
        price: float,
        change_24h: float,
        history: list[float],
        spread: float | None = None,
        id: str | None = None,
    ):
        super().__init__(id=id)
        self._symbol = symbol
        self._price = price
        self._change_24h = change_24h
        self._history = history
        self._spread = spread
        self._logged_update_error = False
        self._last_flash_time: float = 0.0  # Rate limit flashing

    @property
    def symbol(self) -> str:
        """Get the ticker symbol."""
        return self._symbol

    def compose(self) -> ComposeResult:
        # Compact row: Symbol | Sparkline | Price (Change) | Spread
        with Horizontal(classes="ticker-row-inner"):
            yield Label(self._symbol, classes="ticker-symbol", id="ticker-symbol")
            yield SparklineWidget(
                self._history, color_trend=True, classes="ticker-spark", id="ticker-spark"
            )

            price_color = "green" if self._change_24h >= 0 else "red"
            arrow = "↗" if self._change_24h >= 0 else "↘"
            yield Label(
                f"{format_currency(self._price)}", classes="ticker-price", id="ticker-price"
            )
            yield Label(
                f"[{price_color}]{arrow} {self._change_24h:+.1f}%[/]",
                classes="ticker-change",
                id="ticker-change",
            )

            # Spread column
            spread_str = f"{self._spread:.3f}%" if self._spread is not None else "--"
            yield Label(spread_str, classes="ticker-spread", id="ticker-spread")

    def update_values(
        self,
        price: float,
        change_24h: float,
        history: list[float],
        spread: float | None = None,
    ) -> None:
        """Update ticker values in-place without recreating the widget.

        This is a performance optimization - instead of removing and
        remounting the entire TickerRow, we just update the labels.

        Args:
            price: New price value.
            change_24h: New 24h change percentage.
            history: New price history for sparkline.
            spread: New spread percentage (optional).
        """
        # Only update if values actually changed
        if (
            price == self._price
            and change_24h == self._change_24h
            and history == self._history
            and spread == self._spread
        ):
            return

        # Capture old price for flash detection before updating
        old_price = self._price

        self._price = price
        self._change_24h = change_24h
        self._history = history
        self._spread = spread

        try:
            # Update price label with flash effect
            price_label = self.query_one("#ticker-price", Label)
            price_label.update(f"{format_currency(price)}")

            # Flash price if it changed (rate limited to 1 per second)
            import time as time_mod

            now = time_mod.time()
            if old_price != price and (now - self._last_flash_time) > 1.0:
                direction = "up" if price > old_price else "down"
                flash_label(price_label, direction=direction, duration=0.4)
                self._last_flash_time = now

            # Update change label with color
            price_color = "green" if change_24h >= 0 else "red"
            arrow = "↗" if change_24h >= 0 else "↘"
            change_label = self.query_one("#ticker-change", Label)
            change_label.update(f"[{price_color}]{arrow} {change_24h:+.1f}%[/]")

            # Update sparkline
            sparkline = self.query_one("#ticker-spark", SparklineWidget)
            sparkline.data = history

            # Update spread
            spread_label = self.query_one("#ticker-spread", Label)
            spread_str = f"{spread:.3f}%" if spread is not None else "--"
            spread_label.update(spread_str)
        except Exception as e:
            # Avoid spamming logs if we're called before mount.
            if self.is_mounted and not self._logged_update_error:
                logger.debug(
                    "TickerRow update failed for %s: %s",
                    self._symbol,
                    e,
                    exc_info=True,
                )
                self._logged_update_error = True


class MarketPulseWidget(Static):
    """
    Dashboard widget displaying market overview.
    Designed for the 'Market' tile in the Bento Grid.

    Implements StateObserver to receive updates via StateRegistry broadcast.

    OPTIMIZATION: Uses delta updates - existing TickerRow widgets are
    updated in-place rather than being destroyed and recreated on every
    data change. Only adds/removes rows when the symbol list changes.

    Styles are defined in styles/widgets/dashboard.tcss for centralized theming.
    """

    SCOPED_CSS = False  # Use global styles from dashboard.tcss

    # Can accept a list of market data dicts
    market_data = reactive([], always_update=True)  # type: ignore

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cache ticker rows by symbol for delta updates
        self._ticker_cache: dict[str, TickerRow] = {}
        self._has_received_update = False
        self._bot_running = False
        self._data_source_mode = "demo"
        self._connection_status = "UNKNOWN"

    def on_mount(self) -> None:
        """Register with StateRegistry on mount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from StateRegistry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts market data from TuiState.market_data (MarketState)
        and converts to the list format expected by watch_market_data().
        """
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
            # Priority 1: Staleness/degraded banner
            staleness_result = get_staleness_banner(state)
            if staleness_result:
                banner.update_banner(staleness_result[0], severity=staleness_result[1])
            else:
                # Priority 2: Connection-specific banner
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
            logger.debug("MarketPulseWidget banner update failed: %s", e, exc_info=True)

        # Update data state indicator
        self._update_data_state_indicator(state)

        if not state.market_data or not state.market_data.prices:
            # Trigger watch with empty list to show empty state
            self.market_data = []
            return

        market_list = []
        for symbol, price in state.market_data.prices.items():
            history = state.market_data.price_history.get(symbol, [])
            spread = state.market_data.spreads.get(symbol)
            # Calculate change from oldest price in history
            price_float = float(price)
            change_pct = calculate_price_change_percent(price_float, history)

            market_list.append(
                {
                    "symbol": symbol,
                    "price": price_float,
                    "change_24h": change_pct,  # Recent change from oldest price in rolling history
                    "history": [float(h) for h in history[-10:]] if history else [],
                    "spread": float(spread) if spread else None,
                }
            )

        self.market_data = market_list

    def _update_data_state_indicator(self, state: TuiState) -> None:
        """Update data freshness indicator in header with relative time."""
        try:
            indicator = self.query_one("#market-data-state", Label)
            freshness = get_freshness_display(state)

            if freshness:
                text, css_class = freshness
                indicator.update(text)
                # Update CSS classes
                indicator.remove_class("fresh", "stale", "critical")
                indicator.add_class(css_class)
            else:
                indicator.update("")
                indicator.remove_class("fresh", "stale", "critical")
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        with Horizontal(classes="market-header"):
            yield Label("MARKET PULSE", classes="widget-header")
            yield Label("", id="market-data-state", classes="data-state-label")
        yield TileBanner(id="market-banner", classes="tile-banner hidden")
        # Container for rows - starts with loading state
        with Vertical(id="ticker-list"):
            yield tile_loading_state("Waiting for market feed…")

    def watch_market_data(self, data: list) -> None:
        """Update market display with delta logic.

        OPTIMIZATION: Instead of remove_children() + mount() for every update,
        we update existing rows in-place and only add/remove when needed.
        """
        try:
            container = self.query_one("#ticker-list", Vertical)

            # Limit to fit tile
            display_data = data[:6] if data else []

            if not display_data:
                container.remove_children()
                self._ticker_cache.clear()
                if not self._has_received_update:
                    container.mount(tile_loading_state("Waiting for market feed..."))
                    return

                # Use shared empty state config for consistency
                config = get_empty_state_config(
                    data_type="Market",
                    bot_running=self._bot_running,
                    data_source_mode=self._data_source_mode,
                    connection_status=self._connection_status,
                )

                # Handle connecting state with loading spinner
                conn = self._connection_status.upper()
                if conn in ("CONNECTING", "RECONNECTING", "SYNCING", "UNKNOWN", "--", ""):
                    if self._bot_running:
                        container.mount(tile_loading_state("Connecting to market feed..."))
                        return

                container.mount(
                    tile_empty_state(
                        config["title"],
                        config["subtitle"],
                        icon=config["icon"],
                        actions=config["actions"],
                    )
                )
                return

            # Track which symbols we see in this update
            seen_symbols: set[str] = set()

            # Remove any placeholder children (loading/empty) once real data arrives.
            for child in list(container.children):
                if isinstance(child, Container) and "empty-state-container" in child.classes:
                    child.remove()

            for item in display_data:
                symbol = item.get("symbol", "UNKNOWN")
                seen_symbols.add(symbol)

                price = float(item.get("price", 0.0))
                change = float(item.get("change_24h", 0.0))
                history = item.get("history", [])
                spread = item.get("spread")

                if symbol in self._ticker_cache:
                    # DELTA UPDATE: Update existing row in-place
                    self._ticker_cache[symbol].update_values(price, change, history, spread)
                else:
                    # NEW: Create and mount new row
                    row = TickerRow(
                        symbol, price, change, history, spread=spread, id=f"ticker-{symbol}"
                    )
                    self._ticker_cache[symbol] = row
                    container.mount(row)

            # Remove rows for symbols no longer in data
            symbols_to_remove = set(self._ticker_cache.keys()) - seen_symbols
            for symbol in symbols_to_remove:
                row = self._ticker_cache.pop(symbol)
                row.remove()

        except Exception as e:
            # Surface rendering failures instead of silently swallowing them.
            logger.warning("MarketPulseWidget render failed: %s", e, exc_info=True)
            try:
                container = self.query_one("#ticker-list", Vertical)
                container.remove_children()
                container.mount(
                    tile_empty_state(
                        "Market Pulse Error",
                        "Render failed — see logs",
                        icon="⚠",
                        actions=["[R] Reconnect", "[L] Logs"],
                    )
                )
                try:
                    self.query_one(TileBanner).update_banner(
                        "Market tile render error",
                        severity="error",
                    )
                except Exception:
                    pass
            except Exception:
                pass


class PositionCardWidget(Static):
    """
    Hero widget displaying the active position in detail.
    Designed for the top-left 'Hero' tile.

    Implements StateObserver to receive updates via StateRegistry broadcast.

    Styles are defined in styles/widgets/dashboard.tcss for centralized theming.
    """

    SCOPED_CSS = False  # Use global styles from dashboard.tcss

    position_data = reactive(None, always_update=True)  # type: ignore

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

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts position data from TuiState.position_data (PortfolioSummary)
        and converts to the dict format expected by watch_position_data().
        """
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


@dataclass(frozen=True)
class SystemThresholds:
    """Configurable thresholds for system monitor color coding.

    All values define boundaries between OK/WARNING/CRITICAL states.
    Values aligned with shared thresholds in gpt_trader.tui.thresholds.
    """

    # Latency thresholds (milliseconds)
    latency_good: float = 50.0  # Below = OK
    latency_warn: float = 150.0  # Below = WARNING, above = CRITICAL

    # Rate limit thresholds (percentage)
    rate_limit_good: float = 50.0  # Below = OK
    rate_limit_warn: float = 80.0  # Below = WARNING, above = CRITICAL

    # CPU thresholds (percentage)
    cpu_warn: float = 50.0  # Below = OK
    cpu_critical: float = 80.0  # Below = WARNING, above = CRITICAL

    # Memory thresholds (MB)
    memory_max: float = 1024.0  # Max memory for percentage calculation
    memory_warn: float = 60.0  # Percentage threshold for WARNING
    memory_critical: float = 80.0  # Percentage threshold for CRITICAL


# Default thresholds instance
DEFAULT_THRESHOLDS = SystemThresholds()


class SystemMonitorWidget(Static):
    """
    Displays system health metrics in a compact panel.

    Shows: CPU, Memory, Latency, Connection Status, Rate Limit.
    Implements StateObserver to receive updates via StateRegistry broadcast.

    Styles are defined in styles/widgets/dashboard.tcss for centralized theming.

    Args:
        thresholds: Optional SystemThresholds for configurable color coding.
    """

    SCOPED_CSS = False  # Use global styles from dashboard.tcss

    cpu_usage = reactive(0.0)
    memory_usage = reactive("0MB")
    latency = reactive(0.0)
    connection_status = reactive("CONNECTING")
    rate_limit = reactive("0%")

    # Resilience metrics from CoinbaseClient
    latency_p50 = reactive(0.0)
    latency_p95 = reactive(0.0)
    error_rate_pct = reactive(0.0)
    cache_hit_rate_pct = reactive(0.0)
    circuit_state = reactive("OK")

    # Execution telemetry
    exec_success_rate = reactive(100.0)
    exec_latency_ms = reactive(0.0)
    exec_count = reactive(0)

    def __init__(
        self,
        thresholds: SystemThresholds | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def on_mount(self) -> None:
        """Register with StateRegistry on mount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.register(self)

    def on_unmount(self) -> None:
        """Unregister from StateRegistry on unmount."""
        if hasattr(self.app, "state_registry"):
            self.app.state_registry.unregister(self)

    def on_state_updated(self, state: TuiState) -> None:
        """Handle state updates from StateRegistry broadcast.

        Extracts system data from TuiState.system_data (SystemStatus).
        """
        if not state.system_data:
            return

        system_data = state.system_data

        # Extract CPU usage (remove % suffix and convert to float)
        try:
            cpu_val = system_data.cpu_usage
            if isinstance(cpu_val, str) and cpu_val.endswith("%"):
                self.cpu_usage = float(cpu_val.rstrip("%"))
            elif isinstance(cpu_val, (int, float)):
                self.cpu_usage = float(cpu_val)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed to extract CPU usage: %s", e)

        # Extract latency
        try:
            latency_val = system_data.api_latency
            if isinstance(latency_val, (int, float)):
                self.latency = float(latency_val)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed to extract latency: %s", e)

        # Extract memory usage
        try:
            memory_val = system_data.memory_usage
            if memory_val is not None:
                self.memory_usage = str(memory_val)
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract memory usage: %s", e)

        # Extract connection status
        try:
            if state.data_source_mode != "demo" and not state.running:
                self.connection_status = "STOPPED"
            else:
                conn_val = system_data.connection_status
                if conn_val is not None:
                    self.connection_status = str(conn_val)
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract connection status: %s", e)

        # Extract rate limit usage
        try:
            rate_val = system_data.rate_limit_usage
            if rate_val is not None:
                self.rate_limit = str(rate_val)
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract rate limit: %s", e)

        # Extract resilience metrics if available
        try:
            res = state.resilience_data
            if res and res.last_update > 0:
                self.latency_p50 = res.latency_p50_ms
                self.latency_p95 = res.latency_p95_ms
                self.error_rate_pct = res.error_rate * 100
                self.cache_hit_rate_pct = res.cache_hit_rate * 100
                self.circuit_state = "OPEN" if res.any_circuit_open else "OK"
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract resilience metrics: %s", e)

        # Extract execution telemetry if available
        try:
            exec_data = state.execution_data
            if exec_data and exec_data.submissions_total > 0:
                self.exec_success_rate = exec_data.success_rate
                self.exec_latency_ms = exec_data.avg_latency_ms
                self.exec_count = exec_data.submissions_total
        except (TypeError, AttributeError) as e:
            logger.debug("Failed to extract execution metrics: %s", e)

    def compose(self) -> ComposeResult:
        yield Label("SYSTEM", classes="sys-header")

        # CPU with progress bar
        yield ProgressBarWidget(percentage=0.0, label="CPU", id="pb-cpu", classes="sys-row")

        # Memory with progress bar
        yield ProgressBarWidget(percentage=0.0, label="MEM", id="pb-memory", classes="sys-row")

        # Latency with color coding
        yield Label("Latency: 0ms", id="lbl-latency", classes="sys-metric")

        # Connection status
        yield Label("[yellow]○ Connecting...[/yellow]", id="lbl-conn", classes="sys-metric warning")

        # Rate limit with progress bar
        yield ProgressBarWidget(percentage=0.0, label="Rate", id="pb-rate", classes="sys-row")

        # Resilience metrics section
        with Container(id="resilience-section", classes="resilience-metrics"):
            yield Label("p50/p95: --ms / --ms", id="lbl-latency-pct", classes="sys-metric")
            yield Label("Errors: 0.0%", id="lbl-error-rate", classes="sys-metric")
            yield Label("Cache: --%", id="lbl-cache-hit", classes="sys-metric")
            yield Label("[green]Circuit: OK[/green]", id="lbl-circuit", classes="sys-metric")

        # Execution telemetry section
        with Container(id="execution-section", classes="execution-metrics"):
            yield Label("Exec: --% (0)", id="lbl-exec-rate", classes="sys-metric")
            yield Label("Exec Lat: --ms", id="lbl-exec-latency", classes="sys-metric")

    def watch_cpu_usage(self, val: float) -> None:
        try:
            self.query_one("#pb-cpu", ProgressBarWidget).percentage = val
        except Exception as e:
            logger.debug("Failed to update CPU display: %s", e)

    def watch_memory_usage(self, val: str) -> None:
        try:
            pb = self.query_one("#pb-memory", ProgressBarWidget)
            # Extract numeric value from "256MB" format
            try:
                memory_mb = float(val.rstrip("MB").rstrip("GB").rstrip("mb").rstrip("gb"))
                # Convert GB to MB if needed
                if "GB" in val.upper():
                    memory_mb *= 1024
                pct = min(100.0, (memory_mb / self.thresholds.memory_max) * 100)
                pb.percentage = pct
            except (ValueError, AttributeError):
                # If parsing fails, set to 0
                pb.percentage = 0.0
        except Exception as e:
            logger.debug("Failed to update memory display: %s", e)

    def watch_latency(self, val: float) -> None:
        try:
            lbl = self.query_one("#lbl-latency", Label)
            # Color code using shared thresholds with icon for accessibility
            status = get_latency_status(val, PERF_THRESHOLDS)
            color = get_status_color(status)
            icon = get_status_icon(status)
            lbl.update(f"[{color}]{icon} Latency: {val:.0f}ms[/{color}]")
        except Exception as e:
            logger.debug("Failed to update latency display: %s", e)

    def watch_connection_status(self, val: str) -> None:
        try:
            lbl = self.query_one("#lbl-conn", Label)
            status_upper = val.upper()
            if status_upper in ("CONNECTED", "OK", "HEALTHY"):
                lbl.update("[green]● Connected[/green]")
                lbl.remove_class("stopped", "warning", "bad")
                lbl.add_class("good")
                # Flash green when connection is established
                flash_label(lbl, direction="up", duration=0.6)
            elif status_upper in ("STOPPED", "IDLE"):
                lbl.update("[cyan]■ Stopped[/cyan]")
                lbl.remove_class("good", "warning", "bad")
                lbl.add_class("stopped")
            elif status_upper in ("CONNECTING", "RECONNECTING", "SYNCING", "--", "UNKNOWN"):
                lbl.update("[yellow]○ Connecting...[/yellow]")
                lbl.remove_class("stopped", "good", "bad")
                lbl.add_class("warning")
            else:
                lbl.update(f"[red]■ {val}[/red]")
                lbl.remove_class("stopped", "good", "warning")
                lbl.add_class("bad")
                # Flash red when connection has issues
                flash_label(lbl, direction="down", duration=0.6)
        except Exception as e:
            logger.debug("Failed to update connection status display: %s", e)

    def watch_rate_limit(self, val: str) -> None:
        try:
            pb = self.query_one("#pb-rate", ProgressBarWidget)
            # Extract numeric value for progress bar
            try:
                pct = float(val.rstrip("%"))
                pb.percentage = pct
            except (ValueError, AttributeError):
                pb.percentage = 0.0
        except Exception as e:
            logger.debug("Failed to update rate limit display: %s", e)

    def watch_latency_p50(self, val: float) -> None:
        """Update the p50/p95 latency display."""
        try:
            lbl = self.query_one("#lbl-latency-pct", Label)
            lbl.update(f"p50/p95: {val:.0f}ms / {self.latency_p95:.0f}ms")
        except Exception as e:
            logger.debug("Failed to update latency percentiles: %s", e)

    def watch_latency_p95(self, val: float) -> None:
        """Update the p50/p95 latency display when p95 changes."""
        try:
            lbl = self.query_one("#lbl-latency-pct", Label)
            lbl.update(f"p50/p95: {self.latency_p50:.0f}ms / {val:.0f}ms")
        except Exception as e:
            logger.debug("Failed to update latency percentiles: %s", e)

    def watch_error_rate_pct(self, val: float) -> None:
        """Update the error rate display with color coding and icon for accessibility."""
        try:
            lbl = self.query_one("#lbl-error-rate", Label)
            # Color code using shared thresholds with icon for accessibility
            status = get_error_rate_status(val, PERF_THRESHOLDS)
            color = get_status_color(status)
            icon = get_status_icon(status)
            lbl.update(f"[{color}]{icon} Errors: {val:.1f}%[/{color}]")
        except Exception as e:
            logger.debug("Failed to update error rate: %s", e)

    def watch_cache_hit_rate_pct(self, val: float) -> None:
        """Update the cache hit rate display."""
        try:
            lbl = self.query_one("#lbl-cache-hit", Label)
            if val > 0:
                lbl.update(f"Cache: {val:.0f}%")
            else:
                lbl.update("Cache: --%")
        except Exception as e:
            logger.debug("Failed to update cache hit rate: %s", e)

    def watch_circuit_state(self, val: str) -> None:
        """Update the circuit breaker state display."""
        try:
            lbl = self.query_one("#lbl-circuit", Label)
            if val == "OK":
                lbl.update("[green]Circuit: OK[/green]")
            else:
                lbl.update("[red]Circuit: OPEN[/red]")
        except Exception as e:
            logger.debug("Failed to update circuit state: %s", e)

    def watch_exec_success_rate(self, val: float) -> None:
        """Update the execution success rate display."""
        try:
            lbl = self.query_one("#lbl-exec-rate", Label)
            # Color code based on success rate
            if val >= 95:
                color = "green"
                icon = "✓"
            elif val >= 80:
                color = "yellow"
                icon = "●"
            else:
                color = "red"
                icon = "✗"
            lbl.update(f"[{color}]{icon} Exec: {val:.0f}% ({self.exec_count})[/{color}]")
        except Exception as e:
            logger.debug("Failed to update execution rate: %s", e)

    def watch_exec_latency_ms(self, val: float) -> None:
        """Update the execution latency display."""
        try:
            lbl = self.query_one("#lbl-exec-latency", Label)
            if val > 0:
                # Color code based on latency
                if val < 100:
                    color = "green"
                elif val < 500:
                    color = "yellow"
                else:
                    color = "red"
                lbl.update(f"[{color}]Exec Lat: {val:.0f}ms[/{color}]")
            else:
                lbl.update("Exec Lat: --ms")
        except Exception as e:
            logger.debug("Failed to update execution latency: %s", e)

    def watch_exec_count(self, val: int) -> None:
        """Update execution count in the rate display."""
        # Triggers update via watch_exec_success_rate
        try:
            lbl = self.query_one("#lbl-exec-rate", Label)
            rate = self.exec_success_rate
            if rate >= 95:
                color = "green"
                icon = "✓"
            elif rate >= 80:
                color = "yellow"
                icon = "●"
            else:
                color = "red"
                icon = "✗"
            lbl.update(f"[{color}]{icon} Exec: {rate:.0f}% ({val})[/{color}]")
        except Exception as e:
            logger.debug("Failed to update execution count: %s", e)
