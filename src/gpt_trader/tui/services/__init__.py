"""
TUI Services Package.

Services provide focused, single-responsibility functionality extracted
from the main TraderApp to reduce complexity and improve testability.

Services are initialized by TraderApp and handle specific concerns:
- ActionDispatcher: Keyboard shortcut action handling
- ThemeService: Theme management and persistence
- ConfigService: Configuration display and management
- ResponsiveManager: Terminal resize handling and responsive states
- ModeService: Bot mode creation and switching
- PreferencesService: Unified user preferences persistence
- WorkerService: Background worker management for async operations
- UpdateThrottler: Batches high-frequency updates to reduce UI flicker
- AlertManager: Monitors bot state and triggers notifications
- StateRegistry: Broadcast state updates to registered widgets
- TuiPerformanceService: Performance monitoring and metrics collection
- FocusManager: 2D tile navigation and focus management
- OnboardingService: Setup progress tracking and first-run guidance
"""

from gpt_trader.tui.services.action_dispatcher import ActionDispatcher
from gpt_trader.tui.services.alert_manager import AlertManager
from gpt_trader.tui.services.config_service import ConfigService
from gpt_trader.tui.services.credential_validator import CredentialValidator
from gpt_trader.tui.services.focus_manager import FocusManager, TileFocusChanged
from gpt_trader.tui.services.mode_service import ModeService
from gpt_trader.tui.services.onboarding_service import (
    ChecklistItem,
    OnboardingService,
    OnboardingStatus,
    clear_onboarding_service,
    get_onboarding_service,
)
from gpt_trader.tui.services.performance_service import (
    FrameMetrics,
    PerformanceSnapshot,
    TuiPerformanceService,
    clear_tui_performance_service,
    get_tui_performance_service,
    set_tui_performance_service,
)
from gpt_trader.tui.services.preferences_service import (
    PreferencesService,
    get_preferences_service,
)
from gpt_trader.tui.services.responsive_manager import ResponsiveManager
from gpt_trader.tui.services.state_registry import StateRegistry
from gpt_trader.tui.services.theme_service import ThemeService
from gpt_trader.tui.services.trading_stats_service import (
    TIME_WINDOWS,
    TradingStatsService,
    clear_trading_stats_service,
    get_trading_stats_service,
)
from gpt_trader.tui.services.update_throttler import UpdateThrottler
from gpt_trader.tui.services.worker_service import WorkerService

__all__ = [
    "ActionDispatcher",
    "AlertManager",
    "ChecklistItem",
    "ConfigService",
    "CredentialValidator",
    "FocusManager",
    "FrameMetrics",
    "ModeService",
    "OnboardingService",
    "OnboardingStatus",
    "TileFocusChanged",
    "PerformanceSnapshot",
    "PreferencesService",
    "ResponsiveManager",
    "StateRegistry",
    "ThemeService",
    "TIME_WINDOWS",
    "TradingStatsService",
    "TuiPerformanceService",
    "UpdateThrottler",
    "WorkerService",
    "clear_onboarding_service",
    "clear_trading_stats_service",
    "clear_tui_performance_service",
    "get_onboarding_service",
    "get_preferences_service",
    "get_trading_stats_service",
    "get_tui_performance_service",
    "set_tui_performance_service",
]
