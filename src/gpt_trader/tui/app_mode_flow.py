"""Mode flow mixin for TraderApp.

Contains methods related to mode selection, credential validation,
and mode switching workflows.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from gpt_trader.tui.notification_helpers import notify_action
from gpt_trader.tui.services.mode_service import create_bot_for_mode
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp

logger = get_logger(__name__, component="tui")


class TraderAppModeFlowMixin:
    """Mixin providing mode flow methods for TraderApp.

    Methods:
    - _detect_bot_mode: Detect current bot operating mode
    - _switch_to_mode: Switch to a new bot mode
    - _handle_mode_selection: Handle mode selection from screen
    - _show_validation_screen: Show credential validation screen
    - _check_cached_credentials: Check cached credentials validity
    - _cache_credential_validation: Cache successful validation
    - _handle_wizard_result: Handle API setup wizard result
    - _continue_saved_mode_flow: Continue after saved mode validation
    - _finish_saved_mode_setup: Finish saved mode setup
    - _continue_mode_selection_flow: Continue after mode selection validation
    - _finish_mode_selection_setup: Finish mode selection setup
    """

    # Type hints for attributes from TraderApp
    if TYPE_CHECKING:
        bot: Any
        mode_service: Any
        lifecycle_manager: Any
        _demo_scenario: str
        _initial_mode: str | None

        def exit(self) -> None: ...
        def notify(self, message: str, **kwargs: Any) -> None: ...
        def push_screen(self, screen: Any, callback: Any = None) -> None: ...
        def run_worker(self, coro: Any, **kwargs: Any) -> None: ...
        async def _initialize_with_bot(self) -> None: ...

    def _detect_bot_mode(self: TraderApp) -> str:
        """Detect current bot operating mode.

        Delegates to ModeService or BotLifecycleManager.
        """
        if self.bot:
            return self.mode_service.detect_bot_mode(self.bot)
        if self.lifecycle_manager:
            return self.lifecycle_manager.detect_bot_mode()
        return "demo"  # Default fallback

    async def _switch_to_mode(self: TraderApp, target_mode: str) -> bool:
        """Switch to a new bot mode safely.

        Delegates to BotLifecycleManager.
        """
        if self.lifecycle_manager:
            return await self.lifecycle_manager.switch_mode(target_mode)
        return False

    async def _handle_mode_selection(self: TraderApp, selected_mode: str | None) -> None:
        """Handle mode selection from the mode selection screen."""
        if selected_mode is None:
            logger.info("User cancelled mode selection")
            self.exit()
            return

        # Handle setup wizard request from mode selection
        if selected_mode == "setup":
            from gpt_trader.tui.screens.api_setup_wizard import APISetupWizardScreen
            from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

            logger.info("User requested API setup wizard from mode selection")

            def handle_wizard_complete(result: str | None) -> None:
                """Return to mode selection after wizard completes."""
                if result == "verify":
                    self.notify("Credentials saved! Select a mode to continue.", timeout=5)
                self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

            self.push_screen(APISetupWizardScreen(), callback=handle_wizard_complete)
            return

        # Validate credentials for non-demo modes
        if selected_mode != "demo":
            # Use callback-based validation flow (can't use await with push_screen_wait here)
            self._show_validation_screen(
                selected_mode,
                lambda ok: self._continue_mode_selection_flow(selected_mode, ok),
            )
            return

        # Demo mode - proceed directly without validation
        await self._finish_mode_selection_setup(selected_mode)

    def _show_validation_screen(
        self: TraderApp,
        mode: str,
        on_complete: callable,
    ) -> None:
        """Show credential validation screen with callback.

        This method runs validation in a worker and shows results in a modal.

        Args:
            mode: The trading mode to validate for.
            on_complete: Callback with signature (should_proceed: bool) -> None
        """
        from gpt_trader.tui.screens.api_setup_wizard import APISetupWizardScreen
        from gpt_trader.tui.screens.credential_validation_screen import (
            CredentialValidationScreen,
        )
        from gpt_trader.tui.services.credential_validator import CredentialValidator

        async def do_validation() -> None:
            logger.debug("Validating credentials for mode: %s", mode)
            validator = CredentialValidator(self)
            result = await validator.validate_for_mode(mode)

            # Show validation screen and get result via callback
            def handle_validation_result(should_proceed: bool | str | None) -> None:
                if should_proceed == "setup":
                    # User wants to launch the API key setup wizard
                    logger.info("User requested API key setup wizard")
                    self.push_screen(
                        APISetupWizardScreen(),
                        callback=lambda wizard_result: self._handle_wizard_result(
                            mode, wizard_result, on_complete
                        ),
                    )
                elif should_proceed == "retry":
                    # User wants to retry validation (e.g., after fixing env vars)
                    logger.info(f"User requested retry validation for {mode} mode")
                    notify_action(self, "Retrying validation...")
                    self._show_validation_screen(mode, on_complete)
                elif should_proceed:
                    logger.info(f"Credential validation passed for {mode} mode")
                    # Cache successful validation for quick resume on next launch
                    self._cache_credential_validation(mode)
                    on_complete(True)
                else:
                    logger.info(f"User cancelled credential validation for {mode} mode")
                    on_complete(False)

            self.push_screen(
                CredentialValidationScreen(result),
                callback=handle_validation_result,
            )

        self.run_worker(do_validation(), exclusive=True)

    async def _check_cached_credentials(self: TraderApp, mode: str) -> bool:
        """Check if we have valid cached credentials for the mode.

        Returns True if cache is valid and we can skip validation screen.

        Args:
            mode: Trading mode to check cache for.

        Returns:
            True if cached credentials are valid for this mode.
        """
        from gpt_trader.tui.services.credential_validator import CredentialValidator
        from gpt_trader.tui.services.preferences_service import get_preferences_service

        validator = CredentialValidator(self)
        prefs = get_preferences_service()

        # Compute current fingerprint
        current_fp = validator.compute_credential_fingerprint()
        if not current_fp:
            logger.debug("No credential fingerprint available")
            return False

        # Check cache validity
        is_valid = prefs.is_credential_cache_valid(current_fp, mode)
        return is_valid

    def _cache_credential_validation(self: TraderApp, mode: str) -> None:
        """Cache successful credential validation for quick resume.

        Args:
            mode: The mode that was successfully validated.
        """
        from gpt_trader.tui.services.credential_validator import CredentialValidator
        from gpt_trader.tui.services.preferences_service import get_preferences_service

        validator = CredentialValidator(self)
        prefs = get_preferences_service()

        fingerprint = validator.compute_credential_fingerprint()
        if fingerprint:
            # Get existing validation modes and add this one
            cache = prefs.get_credential_cache()
            validation_modes = cache.get("validation_modes", {})
            validation_modes[mode] = True
            prefs.set_credential_cache(fingerprint, validation_modes)
            logger.debug("Cached credential validation for '%s' mode", mode)

    def _handle_wizard_result(
        self: TraderApp,
        mode: str,
        wizard_result: str | None,
        on_complete: callable,
    ) -> None:
        """Handle result from the API setup wizard.

        Args:
            mode: The trading mode being validated.
            wizard_result: Result from wizard ("verify" or None if cancelled).
            on_complete: Original callback to invoke after validation.
        """
        if wizard_result == "verify":
            # User completed wizard - re-run validation
            logger.debug("Re-validating credentials for %s after wizard completion", mode)
            self._show_validation_screen(mode, on_complete)
        else:
            # User cancelled wizard - return to mode selection
            logger.info("User cancelled setup wizard")
            on_complete(False)

    def _continue_saved_mode_flow(self: TraderApp, mode: str, validation_ok: bool) -> None:
        """Continue saved mode flow after credential validation.

        Args:
            mode: The saved mode being restored.
            validation_ok: Whether validation passed and user wants to proceed.
        """
        if validation_ok:
            # Validation passed - proceed with saved mode setup
            asyncio.create_task(self._finish_saved_mode_setup(mode))
        else:
            # Validation failed or user cancelled - show mode selection
            from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

            logger.warning("Saved mode validation failed, showing mode selection")
            self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

    async def _finish_saved_mode_setup(self: TraderApp, mode: str) -> None:
        """Finish setting up a saved mode after validation.

        Args:
            mode: The validated mode to initialize.
        """
        # Show live warning if needed
        if mode == "live":
            should_continue = await self.mode_service.show_live_warning()
            if not should_continue:
                from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

                logger.info("User declined live mode, showing mode selection")
                self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)
                return

        # Create bot for saved mode
        logger.debug("Creating bot for saved mode: %s", mode)
        self.bot = create_bot_for_mode(mode, self._demo_scenario)

        # Initialize with the bot
        await self._initialize_with_bot()

    def _continue_mode_selection_flow(
        self: TraderApp, selected_mode: str, validation_ok: bool
    ) -> None:
        """Continue mode selection flow after credential validation.

        Args:
            selected_mode: The mode selected by user.
            validation_ok: Whether validation passed and user wants to proceed.
        """
        if validation_ok:
            # Validation passed - proceed with mode setup
            asyncio.create_task(self._finish_mode_selection_setup(selected_mode))
        else:
            # Validation failed or user cancelled - return to mode selection
            from gpt_trader.tui.screens.mode_selection import ModeSelectionScreen

            logger.warning("Validation failed, returning to mode selection")
            self.push_screen(ModeSelectionScreen(), callback=self._handle_mode_selection)

    async def _finish_mode_selection_setup(self: TraderApp, selected_mode: str) -> None:
        """Finish mode selection setup after validation.

        Args:
            selected_mode: The validated mode to initialize.
        """
        # Show live warning before creating bot
        if selected_mode == "live":
            should_continue = await self.mode_service.show_live_warning()
            if not should_continue:
                logger.info("User declined to continue in live mode")
                self.exit()
                return

        # Save mode preference for future launches
        self.mode_service.save_mode_preference(selected_mode)

        # Create bot for selected mode using ModeService
        logger.debug("Creating bot for selected mode: %s", selected_mode)
        self.bot = create_bot_for_mode(selected_mode, self._demo_scenario)

        # Initialize with the newly created bot
        await self._initialize_with_bot()
