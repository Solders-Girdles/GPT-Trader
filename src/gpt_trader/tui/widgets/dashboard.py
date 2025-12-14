"""
Dashboard widgets for the High-Fidelity TUI Bento Layout.

All widgets implement StateObserver protocol to receive state updates
via StateRegistry broadcast instead of direct property assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Label, Static

from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.widgets.primitives import ProgressBarWidget, SparklineWidget
from gpt_trader.tui.widgets.tile_states import TileBanner, tile_empty_state, tile_loading_state

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

    @property
    def symbol(self) -> str:
        """Get the ticker symbol."""
        return self._symbol

    def compose(self) -> ComposeResult:
        # Compact row: Symbol | Sparkline | Price (Change) | Spread
        with Horizontal(classes="ticker-row-inner"):
            yield Label(self._symbol, classes="ticker-symbol", id="ticker-symbol")
            yield SparklineWidget(self._history, color_trend=True, classes="ticker-spark", id="ticker-spark")

            price_color = "green" if self._change_24h >= 0 else "red"
            arrow = "↗" if self._change_24h >= 0 else "↘"
            yield Label(f"{format_currency(self._price)}", classes="ticker-price", id="ticker-price")
            yield Label(f"[{price_color}]{arrow} {self._change_24h:+.1f}%[/]", classes="ticker-change", id="ticker-change")

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

        self._price = price
        self._change_24h = change_24h
        self._history = history
        self._spread = spread

        try:
            # Update price label
            price_label = self.query_one("#ticker-price", Label)
            price_label.update(f"{format_currency(price)}")

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
            conn = self._connection_status.upper()

            # If the bot is intentionally stopped (manual start), avoid showing
            # alarming "Broker: UNKNOWN" banners. The empty-state copy already
            # instructs the user to start.
            if not self._bot_running and not state.degraded_mode:
                banner.update_banner("")
            elif state.degraded_mode:
                reason = state.degraded_reason or "Status reporter unavailable"
                banner.update_banner(f"Degraded mode: {reason}", severity="warning")
            elif not conn or conn in ("UNKNOWN", "CONNECTING", "RECONNECTING", "SYNCING", "--"):
                banner.update_banner("")
            elif conn not in ("CONNECTED", "OK", "HEALTHY"):
                severity = "error" if conn in ("DISCONNECTED", "ERROR", "FAILED") else "warning"
                banner.update_banner(f"Broker: {conn}", severity=severity)
            elif not state.connection_healthy:
                banner.update_banner("Market feed stale — reconnecting…", severity="warning")
            elif state.is_data_stale:
                import time
                age = int(time.time() - state.last_data_fetch)
                banner.update_banner(f"Data stale ({age}s)", severity="warning")
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
            market_list.append({
                "symbol": symbol,
                "price": float(price),
                "change_24h": 0.0,  # TODO: Calculate from history if needed
                "history": [float(h) for h in history[-10:]] if history else [],
                "spread": float(spread) if spread else None,
            })

        self.market_data = market_list

    def _update_data_state_indicator(self, state: TuiState) -> None:
        """Update data freshness indicator in header."""
        try:
            indicator = self.query_one("#market-data-state", Label)
            if state.data_fetching:
                indicator.update("[cyan]Fetching...[/cyan]")
            elif state.is_data_stale:
                import time as time_mod
                age = int(time_mod.time() - state.last_data_fetch)
                indicator.update(f"[yellow]Stale ({age}s)[/yellow]")
            elif state.last_data_fetch > 0:
                from datetime import datetime
                dt = datetime.fromtimestamp(state.last_data_fetch)
                indicator.update(f"[dim]{dt.strftime('%H:%M:%S')}[/dim]")
            else:
                indicator.update("")
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
                    container.mount(tile_loading_state("Waiting for market feed…"))
                    return

                if not self._bot_running:
                    if self._data_source_mode == "read_only":
                        container.mount(tile_empty_state("Market Data", "Press [S] to start data feed"))
                    else:
                        container.mount(tile_empty_state("Market Data", "Press [S] to start bot"))
                    return

                conn = self._connection_status.upper()
                if conn in ("DISCONNECTED", "ERROR", "FAILED"):
                    container.mount(
                        tile_empty_state(
                            "Market Data Unavailable",
                            "Broker disconnected — check API credentials",
                            icon="!",
                        )
                    )
                elif conn in ("CONNECTING", "RECONNECTING", "SYNCING", "UNKNOWN", "--", ""):
                    container.mount(tile_loading_state("Connecting to market feed…"))
                else:
                    container.mount(tile_loading_state("Waiting for first tick…"))
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
                    row = TickerRow(symbol, price, change, history, spread=spread, id=f"ticker-{symbol}")
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
                        icon="!",
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
            conn = self._connection_status.upper()

            # When stopped, don't show "Broker: UNKNOWN" banners (manual start).
            if not self._bot_running and not state.degraded_mode:
                banner.update_banner("")
            elif state.degraded_mode:
                reason = state.degraded_reason or "Status reporter unavailable"
                banner.update_banner(f"Degraded mode: {reason}", severity="warning")
            elif not conn or conn in ("UNKNOWN", "CONNECTING", "RECONNECTING", "SYNCING", "--"):
                banner.update_banner("")
            elif conn not in ("CONNECTED", "OK", "HEALTHY"):
                severity = "error" if conn in ("DISCONNECTED", "ERROR", "FAILED") else "warning"
                banner.update_banner(f"Broker: {conn}", severity=severity)
            elif not state.connection_healthy:
                banner.update_banner("Position data stale — reconnecting…", severity="warning")
            elif state.is_data_stale:
                import time
                age = int(time.time() - state.last_data_fetch)
                banner.update_banner(f"Data stale ({age}s)", severity="warning")
            else:
                banner.update_banner("")
        except Exception as e:
            logger.debug("PositionCardWidget banner update failed: %s", e, exc_info=True)

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
                if not self._bot_running:
                    if self._data_source_mode == "read_only":
                        body.mount(tile_empty_state("No Active Position", "Bot is stopped"))
                    else:
                        body.mount(tile_empty_state("No Active Position", "Bot is stopped"))
                    # Add quick actions when bot is stopped
                    body.mount(self._build_quick_actions())
                    return

                conn = self._connection_status.upper()
                if conn in ("DISCONNECTED", "ERROR", "FAILED"):
                    body.mount(
                        tile_empty_state(
                            "Position Data Unavailable",
                            "Broker disconnected — check API credentials",
                            icon="!",
                        )
                    )
                    return

                # Bot is running but no position - show Strategy + Portfolio snapshots
                body.mount(self._build_no_position_dashboard())
                return

            body.mount(self._build_active_state(pos))
        except Exception as e:
            logger.warning("PositionCardWidget render failed: %s", e, exc_info=True)
            try:
                body.remove_children()
                body.mount(
                    tile_empty_state(
                        "Position Tile Error",
                        "Render failed — see logs",
                        icon="!",
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
        return tile_empty_state("No Active Position", "Press [S] to start bot")

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
        root.compose_add_child(Label("System monitoring market conditions", classes="no-pos-subtitle"))

        # Strategy Snapshot section
        strategy_section = Vertical(classes="snapshot-section")
        strategy_section.compose_add_child(Label("STRATEGY", classes="snapshot-header"))

        if hasattr(self.app, "tui_state"):
            decisions = self.app.tui_state.strategy_data.last_decisions
            if decisions:
                for symbol, dec in list(decisions.items())[:2]:
                    color = "green" if dec.action == "BUY" else "red" if dec.action == "SELL" else "yellow"
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
                    Label(f"Equity: {format_currency(state.position_data.equity)}", classes="snapshot-row")
                )

            # Top crypto holdings (excluding USD/USDC)
            holdings = [
                b for b in state.account_data.balances
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
        header_row.compose_add_child(Label(f" {side} {leverage}x ", classes=f"pos-badge {side_cls}"))
        header_row.compose_add_child(Label(f" {symbol} ", classes="pos-symbol"))
        root.compose_add_child(header_row)

        # 2. PnL Hero
        root.compose_add_child(Label(f"[{color}]💰 {pnl_fmt}[/]", classes="pnl-hero"))

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

    All values define boundaries between good/warning/critical states.
    """

    # Latency thresholds (milliseconds)
    latency_good: float = 50.0  # Below this = green
    latency_warn: float = 200.0  # Below this = yellow, above = red

    # Rate limit thresholds (percentage)
    rate_limit_good: float = 50.0  # Below this = green
    rate_limit_warn: float = 80.0  # Below this = yellow, above = red

    # CPU thresholds (percentage)
    cpu_warn: float = 60.0  # Below this = green
    cpu_critical: float = 85.0  # Below this = yellow, above = red

    # Memory thresholds (MB)
    memory_max: float = 1024.0  # Max memory for percentage calculation
    memory_warn: float = 70.0  # Percentage threshold for yellow
    memory_critical: float = 90.0  # Percentage threshold for red


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
            # Color code based on configurable thresholds
            if val < self.thresholds.latency_good:
                lbl.update(f"[green]Latency: {val:.0f}ms[/green]")
            elif val < self.thresholds.latency_warn:
                lbl.update(f"[yellow]Latency: {val:.0f}ms[/yellow]")
            else:
                lbl.update(f"[red]Latency: {val:.0f}ms[/red]")
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
        """Update the error rate display with color coding."""
        try:
            lbl = self.query_one("#lbl-error-rate", Label)
            if val < 1:
                lbl.update(f"[green]Errors: {val:.1f}%[/green]")
            elif val < 5:
                lbl.update(f"[yellow]Errors: {val:.1f}%[/yellow]")
            else:
                lbl.update(f"[red]Errors: {val:.1f}%[/red]")
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
