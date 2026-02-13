"""
Configuration service for managing bot configuration display.

Handles showing the configuration modal and any related
configuration management operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gpt_trader.app.container import get_application_container
from gpt_trader.app.config.bot_config import BotConfig
from gpt_trader.app.runtime.fingerprint import (
    compute_startup_config_fingerprint,
    compare_startup_config_fingerprints,
    load_startup_config_fingerprint,
)
from gpt_trader.tui.events import ConfigChanged, ConfigReloadRequested
from gpt_trader.tui.notification_helpers import notify_error, notify_warning
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from textual.app import App


logger = get_logger(__name__, component="tui")


class ConfigService:
    """Service for managing configuration display and updates.

    Provides methods to show the configuration modal and handle
    configuration-related events.

    Attributes:
        app: Reference to the parent Textual app.
    """

    def __init__(self, app: App) -> None:
        """Initialize the config service.

        Args:
            app: The parent Textual app.
        """
        self.app = app

    def show_config_modal(self, config: Any) -> None:
        """Show the configuration modal.

        Args:
            config: The bot configuration object to display.
        """
        # Validate fingerprint before showing modal
        self._diagnose_startup_config_fingerprint(config)

        # Import at runtime to avoid circular import
        from gpt_trader.tui.widgets.config import ConfigModal

        try:
            self.app.push_screen(ConfigModal(config))
            logger.debug("Config modal opened")
        except Exception as e:
            logger.error(f"Failed to show config modal: {e}", exc_info=True)
            notify_error(self.app, f"Error showing config: {e}")

    def request_reload(self) -> None:
        """Request a configuration reload.

        Posts a ConfigReloadRequested event for handlers to process.
        """
        self.app.post_message(ConfigReloadRequested())
        logger.debug("Config reload requested")

    def notify_config_changed(self, config: Any) -> None:
        """Notify that configuration has changed.

        Posts a ConfigChanged event with the new configuration.

        Args:
            config: The new configuration object.
        """
        self.app.post_message(ConfigChanged(config=config))
        logger.debug("Config changed event posted")

    def _diagnose_startup_config_fingerprint(self, config: Any) -> None:
        """Detect startup config drift by comparing manifests."""

        if not isinstance(config, BotConfig):
            return

        container = get_application_container()
        if container is None:
            return

        fingerprint_path = container.runtime_paths.config_fingerprint_path
        expected = load_startup_config_fingerprint(fingerprint_path)
        if expected is None:
            return

        actual = compute_startup_config_fingerprint(config)
        match, reason = compare_startup_config_fingerprints(expected, actual)
        if match:
            logger.debug(
                "Startup config fingerprint validated (TUI)",
                config_digest=actual.digest,
            )
            return

        notify_warning(
            self.app,
            f"Configuration fingerprint mismatch: {reason}",
            title="Configuration",
            recovery_hint="Restart the CLI/TUI to refresh startup state",
        )
        logger.warning("Config fingerprint mismatch detected via TUI", reason=reason)
