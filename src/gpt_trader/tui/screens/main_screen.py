"""
Main trading dashboard screen.

This screen displays the primary bot monitoring interface with a log-centric layout:
- Strategy decisions at top (what the bot is deciding)
- Logs as primary view (real-time monitoring)
- Market and Portfolio data accessible via overlays
"""

from __future__ import annotations

from textual._context import NoActiveAppError
from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Header

from gpt_trader.tui.events import MainScreenRefreshRequested
from gpt_trader.tui.formatting import format_currency
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    ContextualFooter,
    LogWidget,
    SlimStatusWidget,
    StrategyWidget,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class MainScreen(Screen):
    """Main screen for the GPT-Trader TUI - Log-Centric Bot Monitoring.

    This is the primary dashboard screen optimized for bot monitoring:
    - Slim status bar with essential metrics (1 line)
    - Strategy decisions panel (what the bot is deciding)
    - Logs as primary view (real-time monitoring)
    - Market and Portfolio data accessible via [M] and [D] overlays

    The screen supports responsive layouts and automatic state propagation
    to child widgets.
    """

    # Responsive design property
    responsive_state = reactive(ResponsiveState.STANDARD)

    # Reactive state for state propagation to widgets
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """Propagate state to all registered widgets via StateRegistry."""
        if state is None:
            return

        # Broadcast state to all registered observers via registry
        # Guard against unit tests without active app context
        try:
            if hasattr(self.app, "state_registry"):
                self.app.state_registry.broadcast(state)
        except NoActiveAppError:
            # No app context (e.g., unit tests) - skip broadcasting
            pass

    def compose(self) -> ComposeResult:
        """Compose the log-centric bot monitoring layout.

        Layout structure:
        - Header (1 line)
        - SlimStatusWidget (1 line) - essential metrics
        - Error tracker (hidden when empty)
        - Strategy panel (4-8 lines) - bot decisions
        - Logs (fills remaining) - primary monitoring view
        - Footer (1 line)
        """
        yield Header(show_clock=True, classes="sub-header", icon="*", time_format="%H:%M:%S")

        # Slim status bar with essential metrics
        yield SlimStatusWidget(id="slim-status")

        # Error tracker widget (hidden when no errors)
        yield self.app.error_tracker  # type: ignore[attr-defined]

        # Main Workspace Container - Vertical Log-Centric Layout
        with Container(id="main-workspace"):
            # Strategy at top (fixed height, 4-8 lines)
            yield StrategyWidget(id="dash-strategy", classes="strategy-panel")

            # Logs fill remaining space (primary monitoring view)
            yield LogWidget(id="dash-logs", classes="logs-panel")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Called when screen is mounted - trigger initial data load."""
        logger.info("MainScreen mounted, performing initial UI sync")

        # Initialize slim status widget with current mode
        try:
            slim_status = self.query_one(SlimStatusWidget)
            slim_status.data_source_mode = self.app.data_source_mode  # type: ignore[attr-defined]
            logger.info(f"Initialized SlimStatusWidget: mode={slim_status.data_source_mode}")
        except Exception as e:
            logger.warning(f"Failed to initialize SlimStatusWidget: {e}")

        # Now that all widgets are mounted, we can safely update them
        # Access the app instance to get state and trigger update
        if hasattr(self.app, "_sync_state_from_bot"):
            self.app._sync_state_from_bot()
            self.update_ui(self.app.tui_state)  # type: ignore[attr-defined]
            logger.info("Initial UI sync completed successfully")

        # Initialize responsive state from app
        if hasattr(self.app, "responsive_state"):
            self.responsive_state = self.app.responsive_state  # type: ignore[attr-defined]
            logger.info(f"Initialized responsive state: {self.responsive_state}")

        # Log system ready message for user visibility
        logger.info("=" * 60)
        logger.info("GPT-Trader TUI Ready - Press 'S' to start bot")
        logger.info(f"Mode: {self.app.data_source_mode.upper()}")  # type: ignore[attr-defined]
        logger.info("Keyboard shortcuts: [M] Market | [D] Details | [S] Start/Stop")
        logger.info("=" * 60)

    def update_ui(self, state: TuiState) -> None:
        """Update widgets from TuiState - triggers reactive cascade."""
        # Set state to trigger reactive propagation to all child widgets
        self.state = state

        # Update non-reactive widgets manually (SlimStatusWidget, Footer)

        # Update Footer with data source info
        try:
            footer = self.query_one(ContextualFooter)
            is_healthy = state.check_connection_health()
            footer.update_data_source_info(state.data_source_mode, is_healthy)
        except Exception as e:
            logger.debug(f"Failed to update footer data source info: {e}")

        # Update Slim Status Widget
        try:
            slim_status = self.query_one(SlimStatusWidget)
            slim_status.running = state.running
            slim_status.uptime = state.uptime
            slim_status.data_source_mode = state.data_source_mode
            # Format Decimal values to strings for display
            slim_status.equity = format_currency(state.position_data.equity, decimals=2).replace(
                "$", ""
            )
            slim_status.pnl = format_currency(
                state.position_data.total_unrealized_pnl, decimals=2
            ).replace("$", "")
            # Update position count
            slim_status.position_count = len(state.position_data.positions)
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Failed to update slim status widget: {e}")

        # All other widgets (Strategy) are updated automatically via
        # reactive state propagation (see watch_state above)

    def watch_responsive_state(self, state: ResponsiveState) -> None:
        """Update workspace CSS classes when responsive state changes.

        Applies responsive CSS classes to adjust column ratios based on
        terminal width. Uses batch_update for efficient repainting.

        Args:
            state: ResponsiveState enum value
        """
        try:
            workspace = self.query_one("#main-workspace")

            # Remove all responsive state classes
            workspace.remove_class(
                "workspace--compact",
                "workspace--standard",
                "workspace--comfortable",
                "workspace--wide",
            )

            # Use enum value (not repr) to build valid class names
            state_value = state.value if isinstance(state, ResponsiveState) else str(state)
            # Add new state class
            workspace.add_class(f"workspace--{state_value}")

            logger.debug(f"Workspace CSS updated to: workspace--{state_value}")

        except Exception as e:
            logger.warning(f"Failed to update workspace responsive state: {e}")

        # Propagate state to child widgets
        try:
            slim_status = self.query_one(SlimStatusWidget)
            slim_status.responsive_state = state
            logger.debug(f"Propagated responsive state to SlimStatusWidget: {state}")
        except Exception as e:
            logger.debug(f"Failed to propagate state to SlimStatusWidget: {e}")

        try:
            footer = self.query_one(ContextualFooter)
            footer.responsive_state = state
            logger.debug(f"Propagated responsive state to ContextualFooter: {state}")
        except Exception as e:
            logger.debug(f"Failed to propagate state to ContextualFooter: {e}")

    def on_main_screen_refresh_requested(
        self, event: MainScreenRefreshRequested  # noqa: ARG002
    ) -> None:
        """Handle refresh request from event system.

        This handler replaces direct update_ui() calls from BotLifecycleManager.
        """
        if hasattr(self.app, "tui_state"):
            self.update_ui(self.app.tui_state)  # type: ignore[attr-defined]
            logger.debug("MainScreen refreshed via event")
