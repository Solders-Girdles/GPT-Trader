"""
Main TUI Application for GPT-Trader.

This is the primary entry point for the terminal user interface.
The app coordinates between various services and managers to provide
a complete trading experience.

This module uses mixins to organize functionality:
- TraderAppModeFlowMixin: Mode selection and switching
- TraderAppLifecycleMixin: Mount/unmount lifecycle
- TraderAppBootstrapMixin: Initial data loading
- TraderAppStatusMixin: Status updates and observers
- TraderAppActionsMixin: User actions and event handlers
"""

from __future__ import annotations

import json
import signal
from typing import TYPE_CHECKING, Any

from textual.app import App
from textual.binding import Binding
from textual.command import Provider
from textual.reactive import reactive

from gpt_trader.tui.app_actions import TraderAppActionsMixin
from gpt_trader.tui.app_bootstrap import TraderAppBootstrapMixin
from gpt_trader.tui.app_lifecycle import TraderAppLifecycleMixin
from gpt_trader.tui.app_mode_flow import TraderAppModeFlowMixin
from gpt_trader.tui.app_status import TraderAppStatusMixin
from gpt_trader.tui.commands import TraderCommands
from gpt_trader.tui.preferences_paths import resolve_preferences_paths
from gpt_trader.tui.responsive_state import ResponsiveState
from gpt_trader.tui.services import (
    ActionDispatcher,
    AlertManager,
    ConfigService,
    ModeService,
    ResponsiveManager,
    StateRegistry,
    ThemeService,
    TuiPerformanceService,
    set_tui_performance_service,
)
from gpt_trader.tui.state import TuiState
from gpt_trader.tui.widgets.error_indicator import ErrorIndicatorWidget
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.bot import TradingBot
    from gpt_trader.tui.managers import BotLifecycleManager, UICoordinator
    from gpt_trader.tui.services.worker_service import WorkerService

logger = get_logger(__name__, component="tui")


class TraderApp(
    TraderAppModeFlowMixin,
    TraderAppLifecycleMixin,
    TraderAppBootstrapMixin,
    TraderAppStatusMixin,
    TraderAppActionsMixin,
    App,
):
    """GPT-Trader Terminal User Interface.

    The main application class that coordinates all TUI components.
    Uses a service-oriented architecture to separate concerns:

    - ThemeService: Theme management and persistence
    - ConfigService: Configuration display
    - ResponsiveManager: Terminal resize handling
    - ModeService: Bot mode management
    - BotLifecycleManager: Bot start/stop operations
    - UICoordinator: UI update loop and state synchronization

    Functionality is organized into mixins:
    - TraderAppModeFlowMixin: Mode selection, credential validation, mode switching
    - TraderAppLifecycleMixin: on_mount, on_unmount, on_resize, initialization
    - TraderAppBootstrapMixin: Bootstrap snapshot, read-only data feed
    - TraderAppStatusMixin: Status updates, observer connections, state sync
    - TraderAppActionsMixin: All action_* methods and event handlers
    """

    # Use absolute path resolving relative to this file
    from pathlib import Path

    CSS_PATH = Path(__file__).parent / "styles" / "main.tcss"

    # Enable command palette with custom commands
    COMMANDS: set[type[Provider]] = {TraderCommands}

    # Responsive design properties
    terminal_width = reactive(120)
    responsive_state = reactive(ResponsiveState.STANDARD)

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("s", "toggle_bot", "Start/Stop Bot"),
        ("c", "show_config", "Config"),
        ("l", "focus_logs", "Focus Logs"),
        ("m", "show_market", "Market"),  # Changed from mode_info
        ("d", "show_details", "Details"),  # New: Details overlay
        ("i", "show_mode_info", "Mode Info"),  # Moved from 'm' to 'i'
        ("r", "reconnect_data", "Reconnect"),
        ("p", "panic", "PANIC"),
        ("t", "toggle_theme", "Toggle Theme"),
        ("a", "show_alerts", "Alerts"),
        ("f", "force_refresh", "Refresh"),
        ("y", "show_strategy", "Strategy"),
        ("1", "show_full_logs", "Full Logs"),
        ("2", "show_system_details", "System"),
        ("e", "show_exec_issues", "Exec Issues"),
        ("?", "show_help", "Help"),
        # Command palette (Ctrl+K is more intuitive than default Ctrl+\)
        Binding("ctrl+k", "command_palette", "Commands", show=True),
        # Performance monitoring (Ctrl+P)
        Binding("ctrl+p", "toggle_performance", "Performance", show=False),
        # Log level shortcuts (hidden from footer, shown in help)
        Binding("ctrl+1", "set_log_level('DEBUG')", "Log: DEBUG", show=False),
        Binding("ctrl+2", "set_log_level('INFO')", "Log: INFO", show=False),
        Binding("ctrl+3", "set_log_level('WARNING')", "Log: WARN", show=False),
        Binding("ctrl+4", "set_log_level('ERROR')", "Log: ERROR", show=False),
    ]

    def __init__(
        self,
        bot: TradingBot | Any | None = None,
        initial_mode: str | None = None,
        demo_scenario: str = "mixed",
    ) -> None:
        """Initialize the TUI application.

        Args:
            bot: Optional bot instance (for backward compatibility with run command)
            initial_mode: Optional mode to start in. If None, shows mode selection screen
            demo_scenario: Scenario to use for demo mode
        """
        # Select CSS based on saved theme preference before App init.
        from gpt_trader.tui.theme import ThemeMode

        theme_mode = ThemeMode.DARK
        preferences_path, fallback_path = resolve_preferences_paths()
        for path in filter(None, (preferences_path, fallback_path)):
            try:
                if path.exists():
                    prefs = json.loads(path.read_text())
                    theme_mode = ThemeMode(prefs.get("theme", "dark"))
                    break
            except Exception:
                theme_mode = ThemeMode.DARK

        styles_dir = self.Path(__file__).parent / "styles"
        css_file = styles_dir / (
            "main_light.tcss" if theme_mode == ThemeMode.LIGHT else "main.tcss"
        )
        if not css_file.exists():
            # Prefer a functional UI over a hard crash if the light CSS hasn't
            # been generated yet.
            css_file = styles_dir / "main.tcss"

        super().__init__(css_path=css_file)
        self.bot = bot  # May be None if using mode selection flow
        self.data_source_mode: str = "demo"  # Will be set during on_mount
        self._initial_mode = initial_mode  # For mode selection flow
        self._demo_scenario = demo_scenario  # For demo mode

        # Initialize State
        self.tui_state: TuiState = TuiState()

        # Initialize error tracker widget (singleton for whole app)
        self.error_tracker: ErrorIndicatorWidget = ErrorIndicatorWidget(max_errors=10)

        # Startup bootstrap: fetch balances/market snapshot even while STOPPED.
        self._bootstrap_snapshot_requested: bool = False
        self._bootstrap_snapshot_inflight: bool = False

        # Initialize services
        self.theme_service = ThemeService(self)
        self.config_service = ConfigService(self)
        self.responsive_manager = ResponsiveManager(self)
        self.mode_service = ModeService(self, demo_scenario=demo_scenario)
        self.action_dispatcher = ActionDispatcher(self)
        self.alert_manager = AlertManager(self)
        self.state_registry = StateRegistry()

        # Initialize performance monitoring service
        self.performance_service = TuiPerformanceService(app=self, enabled=True)
        set_tui_performance_service(self.performance_service)

        # Load saved theme preference
        self.theme_service.load_preference()

        # Create managers (only after bot is set)
        self.lifecycle_manager: BotLifecycleManager | None = None
        self.ui_coordinator: UICoordinator | None = None
        self.worker_service: WorkerService | None = None

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def apply_theme_css(self, theme_mode: Any) -> bool:
        """Hot-swap the active CSS file for the requested theme mode."""
        from gpt_trader.tui.theme import ThemeMode

        mode = theme_mode if isinstance(theme_mode, ThemeMode) else ThemeMode(str(theme_mode))
        styles_dir = self.Path(__file__).parent / "styles"

        # Map theme modes to CSS files
        css_file_map = {
            ThemeMode.DARK: "main.tcss",
            ThemeMode.LIGHT: "main_light.tcss",
            ThemeMode.HIGH_CONTRAST: "main_high_contrast.tcss",
        }
        css_file = styles_dir / css_file_map.get(mode, "main.tcss")
        if not css_file.exists():
            return False

        theme_css_paths = {
            str((styles_dir / "main.tcss").absolute()),
            str((styles_dir / "main_light.tcss").absolute()),
            str((styles_dir / "main_high_contrast.tcss").absolute()),
        }

        # Remove any previously-loaded theme CSS sources so switching doesn't
        # accumulate duplicate rules over time.
        for key in list(self.stylesheet.source.keys()):
            path, _scope = key
            if path in theme_css_paths:
                self.stylesheet.source.pop(key, None)

        self.stylesheet.read(css_file)
        self.refresh_css(animate=False)
        return True

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle termination signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        # Schedule the exit on the event loop
        if hasattr(self, "exit"):
            self.exit()
