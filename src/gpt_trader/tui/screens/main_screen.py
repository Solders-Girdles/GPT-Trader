"""
Main trading dashboard screen.

This screen displays the primary bot monitoring interface with a Bento Grid layout:
- Top: CommandBar (Header)
- Middle: Dashboard Grid (Position Hero, Account, Market Pulse)
- Bottom: Console (Logs)

Supports 2D tile navigation via arrow keys and Enter to drill-down.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Grid, Horizontal
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Label

from gpt_trader.tui.events import MainScreenRefreshRequested
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.services.focus_manager import FocusManager, TileFocusChanged
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    ContextualFooter,
    LogWidget,
)
from gpt_trader.tui.widgets.account import AccountWidget
from gpt_trader.tui.widgets.dashboard import (
    MarketPulseWidget,
    PositionCardWidget,
    SystemMonitorWidget,
)
from gpt_trader.tui.widgets.shell import CommandBar
from gpt_trader.tui.widgets.strategy_performance import StrategyPerformanceWidget
from gpt_trader.tui.widgets.trading_stats import TradingStatsWidget
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class MainScreen(Screen):
    """Main screen for the GPT-Trader TUI - High Fidelity Dashboard.

    Layout: Bento Grid
    - Header: 3 lines (CommandBar)
    - Hero: Active Position (Top Left)
    - Account: Account Summary (Top Right)
    - Market: Market Pulse (Middle Right)
    - Logs: Console (Bottom)

    Supports 2D tile navigation via arrow keys and Enter to drill-down.
    """

    BINDINGS = [
        # Tile navigation (hidden from footer, documented in help)
        Binding("up", "focus_up", "Focus Up", show=False),
        Binding("down", "focus_down", "Focus Down", show=False),
        Binding("left", "focus_left", "Focus Left", show=False),
        Binding("right", "focus_right", "Focus Right", show=False),
        Binding("enter", "tile_action", "Open Detail", show=False),
        # Trading stats window toggle (shown in tile hints when account focused)
        Binding("w", "cycle_stats_window", "Window", show=False),
        Binding("W", "reset_stats_window", "All", show=False),
    ]

    # Responsive design property
    responsive_state = reactive(ResponsiveState.STANDARD)

    # Reactive state for state propagation to widgets
    state = reactive(None)  # Type: TuiState | None

    def __init__(self, **kwargs) -> None:
        """Initialize MainScreen with FocusManager."""
        super().__init__(**kwargs)
        self._focus_manager: FocusManager | None = None

    def watch_state(self, state: TuiState | None) -> None:
        """Handle state changes.

        Note: StateRegistry.broadcast() is called by UICoordinator.update_main_screen()
        to ensure consistent update timing. This watcher is kept for reactive property
        semantics but does not broadcast to avoid double updates.
        """
        # Broadcast is handled by UICoordinator.update_main_screen() to avoid
        # double broadcasts when TuiState is mutated in-place (same object reference).
        pass

    def compose(self) -> ComposeResult:
        """Compose the Bento Grid layout."""
        # 1. Top Command Bar (Header Area)
        yield CommandBar(
            bot_mode=getattr(self.app, "data_source_mode", "DEMO").upper(), id="header-bar"
        )

        # 2. Main Grid Area
        with Grid(id="bento-grid"):
            # -- Dashboard Area --

            # Hero: Active Position
            with Container(id="tile-hero", classes="bento-tile"):
                yield PositionCardWidget(id="dash-position")
                yield Horizontal(id="hints-hero", classes="tile-actions-hint")

            # Tile: Account Summary + Trading Stats
            with Container(id="tile-account", classes="bento-tile"):
                yield AccountWidget(compact_mode=False, id="dash-account")
                yield TradingStatsWidget(compact=True, id="dash-trading-stats")
                yield Horizontal(id="hints-account", classes="tile-actions-hint")

            # Tile: Market Pulse
            with Container(id="tile-market", classes="bento-tile"):
                yield MarketPulseWidget(id="dash-market")
                yield Horizontal(id="hints-market", classes="tile-actions-hint")

            # Tile: Strategy Performance
            with Container(id="tile-strategy", classes="bento-tile"):
                yield StrategyPerformanceWidget(id="dash-strategy")
                yield Horizontal(id="hints-strategy", classes="tile-actions-hint")

            # Tile: System Monitor (Small strip)
            with Container(id="tile-system", classes="bento-tile"):
                yield SystemMonitorWidget(id="dash-system")
                yield Horizontal(id="hints-system", classes="tile-actions-hint")

            # -- Console Area --
            with Container(id="tile-logs", classes="bento-tile"):
                yield LogWidget(id="dash-logs", classes="logs-panel")
                yield Horizontal(id="hints-logs", classes="tile-actions-hint")

        # 3. Footer
        yield ContextualFooter()

    def on_mount(self) -> None:
        """Called when screen is mounted - trigger initial data load.

        This is the safe point to connect the StatusReporter observer since
        all widgets are now mounted and ready to receive updates.
        """
        logger.debug("MainScreen mounted (Bento Layout), performing initial UI sync")

        # Ensure the global error indicator is visible on the main dashboard.
        # It will remain attached to this screen (and continue collecting errors
        # globally via safe_update) even while overlay screens are pushed.
        try:
            error_tracker = getattr(self.app, "error_tracker", None)
            if error_tracker is not None and getattr(error_tracker, "parent", None) is None:
                self.mount(error_tracker, after="#header-bar")
        except Exception:
            pass

        # Connect StatusReporter observer now that widgets are mounted
        # This prevents race condition where updates arrive before widgets exist
        if hasattr(self.app, "connect_status_observer"):
            self.app.connect_status_observer()
            logger.debug("StatusReporter observer connected - widgets ready for updates")

        # Now that all widgets are mounted, we can safely update them
        if hasattr(self.app, "_sync_state_from_bot"):
            self.app._sync_state_from_bot()
            if hasattr(self.app, "tui_state"):
                self.update_ui(self.app.tui_state)  # type: ignore
            logger.debug("Initial UI sync completed successfully")

        # Populate balances / initial market snapshot even while STOPPED.
        try:
            request_bootstrap = getattr(self.app, "request_bootstrap_snapshot", None)
            if callable(request_bootstrap):
                request_bootstrap()
        except Exception:
            pass

        # Initialize responsive state from app
        if hasattr(self.app, "responsive_state"):
            self.responsive_state = self.app.responsive_state  # type: ignore

        # Log system ready message with mode information
        mode = (
            getattr(self.app, "data_source_mode", "unknown")
            if hasattr(self.app, "data_source_mode")
            else "unknown"
        )
        logger.info(f"GPT-Trader High-Fidelity TUI Ready - Mode: {mode.upper()}")

        # Initialize focus manager for 2D tile navigation
        self._focus_manager = FocusManager(self.app)
        self._focus_manager.enable()
        logger.debug("FocusManager initialized for tile navigation")

    def update_ui(self, state: TuiState) -> None:
        """Update widgets from TuiState - triggers reactive cascade.

        Setting self.state triggers watch_state() which broadcasts to all
        widgets registered with StateRegistry. Individual widgets (MarketPulseWidget,
        PositionCardWidget, SystemMonitorWidget, AccountWidget) receive updates
        via their on_state_updated() methods.

        Only the footer is updated directly here since it's not a StateObserver.
        """
        self.state = state  # Triggers watch_state() -> StateRegistry.broadcast()

    def watch_responsive_state(self, state: ResponsiveState) -> None:
        """Update workspace CSS classes when responsive state changes.

        Toggles responsive CSS classes on the Bento Grid to adapt layout
        based on terminal width. Classes correspond to styles defined in
        styles/layout/workspace.tcss.
        """
        try:
            grid = self.query_one("#bento-grid", Grid)
            # Remove all responsive classes
            grid.remove_class("compact", "standard", "comfortable", "wide")
            # Add current state class (ResponsiveState.value is lowercase string)
            grid.add_class(state.value)
            logger.debug(f"Updated grid responsive class to: {state.value}")
        except Exception as e:
            logger.debug(f"Could not update responsive class: {e}")

    def on_main_screen_refresh_requested(
        self, event: MainScreenRefreshRequested  # noqa: ARG002
    ) -> None:
        """Handle refresh request from event system."""
        if hasattr(self.app, "tui_state"):
            self.update_ui(self.app.tui_state)  # type: ignore[attr-defined]
            logger.debug("MainScreen refreshed via event")

    # === Focus Navigation Actions ===

    def action_focus_up(self) -> None:
        """Move focus to tile above."""
        if self._focus_manager:
            self._focus_manager.move("up")

    def action_focus_down(self) -> None:
        """Move focus to tile below."""
        if self._focus_manager:
            self._focus_manager.move("down")

    def action_focus_left(self) -> None:
        """Move focus to tile on the left."""
        if self._focus_manager:
            self._focus_manager.move("left")

    def action_focus_right(self) -> None:
        """Move focus to tile on the right."""
        if self._focus_manager:
            self._focus_manager.move("right")

    def action_tile_action(self) -> None:
        """Open drill-down screen for the currently focused tile."""
        if not self._focus_manager:
            return

        tile_id = self._focus_manager.current_tile_id
        self._open_tile_detail(tile_id)

    def _open_tile_detail(self, tile_id: str) -> None:
        """Open the appropriate detail screen for a tile.

        Args:
            tile_id: ID of the tile to open detail screen for.
        """
        # Import screens here to avoid circular imports
        from gpt_trader.tui.screens.account_detail_screen import AccountDetailScreen
        from gpt_trader.tui.screens.full_logs_screen import FullLogsScreen
        from gpt_trader.tui.screens.market_detail_screen import MarketDetailScreen
        from gpt_trader.tui.screens.position_detail_screen import PositionDetailScreen
        from gpt_trader.tui.screens.system_details_screen import SystemDetailsScreen

        screen_map = {
            "tile-hero": PositionDetailScreen,  # Position/Strategy detail (enhanced)
            "tile-account": AccountDetailScreen,  # Account detail (enhanced)
            "tile-market": MarketDetailScreen,  # Market detail (enhanced)
            "tile-system": SystemDetailsScreen,  # System details
            "tile-logs": FullLogsScreen,  # Full logs
        }

        screen_class = screen_map.get(tile_id)
        if screen_class:
            logger.debug(f"Opening detail screen for {tile_id}: {screen_class.__name__}")
            self.app.push_screen(screen_class())
        else:
            logger.warning(f"No detail screen mapped for tile: {tile_id}")

    def on_tile_focus_changed(self, event: TileFocusChanged) -> None:
        """Handle tile focus change events.

        Updates the action hints row in the focused tile with relevant shortcuts.

        Args:
            event: The focus change event with tile_id and actions.
        """
        logger.debug(f"Tile focus changed to: {event.tile_id}")

        # Map tile IDs to their hint container IDs
        hint_map = {
            "tile-hero": "hints-hero",
            "tile-account": "hints-account",
            "tile-market": "hints-market",
            "tile-strategy": "hints-strategy",
            "tile-system": "hints-system",
            "tile-logs": "hints-logs",
        }

        # Update the focused tile's hints
        hint_id = hint_map.get(event.tile_id)
        if hint_id:
            try:
                hint_container = self.query_one(f"#{hint_id}", Horizontal)
                hint_container.remove_children()

                # Add action hints as Labels
                for key, description in event.actions:
                    # Format: [Key] Description
                    hint_container.mount(Label(f"[{key}] {description}", classes="action-hint"))
            except Exception as e:
                logger.debug(f"Could not update hints for {event.tile_id}: {e}")

    # === Trading Stats Window Actions ===

    def action_cycle_stats_window(self) -> None:
        """Cycle trading stats time window."""
        try:
            stats_widget = self.query_one("#dash-trading-stats", TradingStatsWidget)
            stats_widget.action_cycle_window()
        except Exception as e:
            logger.debug(f"Could not cycle stats window: {e}")

    def action_reset_stats_window(self) -> None:
        """Reset trading stats to 'All Session' window."""
        try:
            stats_widget = self.query_one("#dash-trading-stats", TradingStatsWidget)
            stats_widget.action_reset_window()
        except Exception as e:
            logger.debug(f"Could not reset stats window: {e}")
