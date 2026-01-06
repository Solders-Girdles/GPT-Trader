"""
Market-related dashboard widgets.

Contains:
- calculate_price_change_percent: Helper for price change calculation
- TickerRow: Single row representing a market ticker
- MarketPulseWidget: Dashboard widget displaying market overview
"""

from __future__ import annotations

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
from gpt_trader.tui.widgets.primitives import SparklineWidget
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
    market_data = reactive([], always_update=True)  # type: ignore[var-annotated]

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
