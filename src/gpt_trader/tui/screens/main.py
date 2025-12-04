from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.screen import Screen
from textual.widgets import Footer, Header, Label

from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets import (
    AccountWidget,
    BotStatusWidget,
    ContextualFooter,
    ExecutionWidget,
    LogWidget,
    MarketWatchWidget,
    ModeIndicator,
    ModeSelector,
    PositionsWidget,
    StrategyWidget,
)
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class MainScreen(Screen):
    """Main screen for the GPT-Trader TUI."""

    # Responsive design property
    responsive_state = reactive("standard")

    # Reactive state for state propagation to widgets
    state = reactive(None)  # Type: TuiState | None

    def watch_state(self, state: TuiState | None) -> None:
        """Propagate state to all child widgets with reactive state."""
        if state is None:
            return

        # Propagate to all widgets that have reactive state
        for widget in self.query("PositionsWidget, ExecutionWidget, MarketWatchWidget, StrategyWidget"):
            if hasattr(widget, "state"):
                widget.state = state  # type: ignore[attr-defined]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        yield BotStatusWidget(id="bot-status-header")

        # Main Workspace Container - Now horizontal split (30/70)
        with Container(id="main-workspace"):
            # Left Column: Market + Strategy (30% width)
            with Container(id="market-strategy-column"):
                yield MarketWatchWidget(id="dash-market", classes="dashboard-item")
                yield StrategyWidget(id="dash-strategy", classes="dashboard-item")

            # Right Column: Execution + Monitoring (70% width)
            with Container(id="execution-monitoring-column"):
                # Positions (40% of right column = 28% of total screen - LARGEST)
                yield PositionsWidget(id="dash-positions", classes="dashboard-item")

                # Execution (Orders + Trades tabbed, 35% combined)
                yield ExecutionWidget(id="dash-execution", classes="dashboard-item")

                # Monitoring Row (20% of right column)
                with Container(id="monitoring-row"):
                    # Logs now get full monitoring row height
                    yield LogWidget(id="dash-logs", classes="dashboard-item compact-logs")

        yield ContextualFooter()

    def on_mount(self) -> None:
        """Called when screen is mounted - trigger initial data load."""
        logger.info("MainScreen mounted, performing initial UI sync")

        # Initialize mode indicator with current mode
        try:
            mode_indicator = self.query_one(ModeIndicator)
            mode_indicator.mode = self.app.data_source_mode  # type: ignore[attr-defined]
            logger.info(f"Initialized ModeIndicator: mode={mode_indicator.mode}")
        except Exception as e:
            logger.warning(f"Failed to initialize ModeIndicator: {e}")

        # Initialize mode selector with current mode
        try:
            mode_selector = self.query_one(ModeSelector)
            mode_selector.current_mode = self.app.data_source_mode  # type: ignore[attr-defined]
            mode_selector.enabled = not self.app.bot.running  # type: ignore[attr-defined]
            logger.info(f"Initialized ModeSelector: mode={mode_selector.current_mode}, enabled={mode_selector.enabled}")
        except Exception as e:
            logger.warning(f"Failed to initialize ModeSelector: {e}")

        # Now that all widgets are mounted, we can safely update them
        # Access the app instance to get state and trigger update
        if hasattr(self.app, "_sync_state_from_bot"):
            self.app._sync_state_from_bot()
            self.update_ui(self.app.tui_state)  # type: ignore[attr-defined]
            if hasattr(self.app, "_pulse_heartbeat"):
                self.app._pulse_heartbeat()
            logger.info("Initial UI sync completed successfully")

        # Initialize responsive state from app
        if hasattr(self.app, "responsive_state"):
            self.responsive_state = self.app.responsive_state  # type: ignore[attr-defined]
            logger.info(f"Initialized responsive state: {self.responsive_state}")

        # Log system ready message for user visibility
        logger.info("=" * 60)
        logger.info("GPT-Trader TUI Ready - Press 'S' to start bot")
        logger.info(f"Mode: {self.app.data_source_mode.upper()}")  # type: ignore[attr-defined]
        logger.info("Keyboard shortcuts: [L] Focus Logs | [1] Full Logs | [2] System Details")
        logger.info("=" * 60)

    def update_ui(self, state: TuiState) -> None:
        """Update widgets from TuiState - triggers reactive cascade."""
        # Set state to trigger reactive propagation to all child widgets
        self.state = state

        # Update non-reactive widgets manually (BotStatusWidget, ModeIndicator, Footer)

        # Update Mode Indicator
        try:
            mode_indicator = self.query_one(ModeIndicator)
            mode_indicator.mode = state.data_source_mode
        except Exception as e:
            logger.debug(f"Failed to update mode indicator: {e}")

        # Update Footer with data source info
        try:
            footer = self.query_one(ContextualFooter)
            is_healthy = state.check_connection_health()
            footer.update_data_source_info(state.data_source_mode, is_healthy)
        except Exception as e:
            logger.debug(f"Failed to update footer data source info: {e}")

        # Update Status (including system health)
        try:
            status_widget = self.query_one(BotStatusWidget)
            status_widget.running = state.running
            status_widget.uptime = state.uptime
            status_widget.equity = state.position_data.equity
            status_widget.pnl = state.position_data.total_unrealized_pnl
            # Update system health in status bar
            status_widget.connection_status = state.system_data.connection_status
            status_widget.api_latency = state.system_data.api_latency
            status_widget.cpu_usage = state.system_data.cpu_usage
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(f"Failed to update bot status widget: {e}")

        # All other widgets (Positions, Execution, Market, Strategy) are updated
        # automatically via reactive state propagation (see watch_state above)

    def watch_responsive_state(self, state: str) -> None:
        """Update workspace CSS classes when responsive state changes.

        Applies responsive CSS classes to adjust column ratios based on
        terminal width. Uses batch_update for efficient repainting.

        Args:
            state: New responsive state ("compact", "standard", "comfortable", "wide")
        """
        try:
            workspace = self.query_one("#main-workspace")

            # Use batch update for efficient CSS class changes
            with self.batch_update():
                # Remove all responsive state classes
                workspace.remove_class(
                    "workspace--compact",
                    "workspace--standard",
                    "workspace--comfortable",
                    "workspace--wide"
                )
                # Add new state class
                workspace.add_class(f"workspace--{state}")

            logger.debug(f"Workspace CSS updated to: workspace--{state}")

        except Exception as e:
            logger.warning(f"Failed to update workspace responsive state: {e}")

        # Propagate state to child widgets
        try:
            status_widget = self.query_one(BotStatusWidget)
            status_widget.responsive_state = state
            logger.debug(f"Propagated responsive state to BotStatusWidget: {state}")
        except Exception as e:
            logger.debug(f"Failed to propagate state to BotStatusWidget: {e}")

        try:
            footer = self.query_one(ContextualFooter)
            footer.responsive_state = state
            logger.debug(f"Propagated responsive state to ContextualFooter: {state}")
        except Exception as e:
            logger.debug(f"Failed to propagate state to ContextualFooter: {e}")


class FullLogsScreen(Screen):
    """Full-screen log viewer with filtering and expanded view."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("📋 FULL SYSTEM LOGS", classes="header")
        yield LogWidget(id="full-logs", compact_mode=False)  # Expanded mode for full logs screen
        yield Footer()

    def action_dismiss(self) -> None:
        """Close the full logs screen and return to main view."""
        self.app.pop_screen()


class SystemDetailsScreen(Screen):
    """Detailed system health and diagnostics screen."""

    BINDINGS = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def compose(self) -> ComposeResult:
        from gpt_trader.tui.widgets.system import SystemHealthWidget

        yield Header(show_clock=True)
        yield Label("⚙️ SYSTEM DETAILS", classes="header")
        with Container(id="system-details-container"):
            yield SystemHealthWidget(
                id="detailed-system", compact_mode=False, classes="dashboard-item"
            )
            yield AccountWidget(id="detailed-account", compact_mode=False, classes="dashboard-item")
        yield Footer()

    def action_dismiss(self) -> None:
        """Close the system details screen and return to main view."""
        self.app.pop_screen()
