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
- WorkerService: Background worker management for async operations
- UpdateThrottler: Batches high-frequency updates to reduce UI flicker
- AlertManager: Monitors bot state and triggers notifications
- StateRegistry: Broadcast state updates to registered widgets
"""

from gpt_trader.tui.services.action_dispatcher import ActionDispatcher
from gpt_trader.tui.services.alert_manager import AlertManager
from gpt_trader.tui.services.config_service import ConfigService
from gpt_trader.tui.services.credential_validator import CredentialValidator
from gpt_trader.tui.services.mode_service import ModeService
from gpt_trader.tui.services.responsive_manager import ResponsiveManager
from gpt_trader.tui.services.state_registry import StateRegistry
from gpt_trader.tui.services.theme_service import ThemeService
from gpt_trader.tui.services.update_throttler import UpdateThrottler
from gpt_trader.tui.services.worker_service import WorkerService

__all__ = [
    "ActionDispatcher",
    "AlertManager",
    "ConfigService",
    "CredentialValidator",
    "ModeService",
    "ResponsiveManager",
    "StateRegistry",
    "ThemeService",
    "UpdateThrottler",
    "WorkerService",
]
