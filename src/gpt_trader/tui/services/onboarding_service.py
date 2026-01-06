"""
Onboarding Status Service.

Tracks setup progress and determines when the system is ready for trading.
Provides checklist state for first-run guidance and "ready" badge status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gpt_trader.tui.services.preferences_service import get_preferences_service
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


@dataclass
class ChecklistItem:
    """A single onboarding checklist item."""

    id: str
    label: str
    description: str
    completed: bool = False
    required: bool = True  # Required for "ready" status
    skippable: bool = False  # Can be skipped in demo mode


@dataclass
class OnboardingStatus:
    """Overall onboarding status with checklist items."""

    items: list[ChecklistItem] = field(default_factory=list)
    mode: str = "demo"

    @property
    def completed_count(self) -> int:
        """Count of completed items."""
        return sum(1 for item in self.items if item.completed)

    @property
    def required_count(self) -> int:
        """Count of required items for current mode."""
        return sum(1 for item in self.items if item.required and not item.skippable)

    @property
    def required_completed(self) -> int:
        """Count of completed required items."""
        return sum(1 for item in self.items if item.required and item.completed)

    @property
    def is_ready(self) -> bool:
        """Check if all required items are complete."""
        for item in self.items:
            if item.required and not item.completed:
                return False
        return True

    @property
    def progress_pct(self) -> float:
        """Progress percentage (0.0 to 1.0)."""
        if not self.items:
            return 1.0
        return self.completed_count / len(self.items)

    @property
    def ready_label(self) -> str:
        """Get human-readable ready status."""
        if self.is_ready:
            return "Ready"
        return f"{self.required_completed}/{self.required_count}"

    def get_next_step(self) -> ChecklistItem | None:
        """Get the next incomplete required item."""
        for item in self.items:
            if item.required and not item.completed:
                return item
        return None


class OnboardingService:
    """Service for tracking and computing onboarding status.

    Evaluates setup progress based on:
    - Trading mode selection
    - API credential validation (for non-demo modes)
    - Connection establishment
    - Risk settings acknowledgment
    """

    def __init__(self) -> None:
        """Initialize the onboarding service."""
        self._preferences = get_preferences_service()

    def get_status(self, state: TuiState | None = None) -> OnboardingStatus:
        """Compute current onboarding status.

        Args:
            state: Current TUI state for connection info.

        Returns:
            OnboardingStatus with checklist items.
        """
        mode = state.data_source_mode if state else "demo"
        is_demo = mode == "demo"

        items = [
            ChecklistItem(
                id="mode_selected",
                label="Mode Selected",
                description=f"Trading mode: {mode.upper()}",
                completed=True,  # Always true if we're running
                required=True,
            ),
            ChecklistItem(
                id="credentials_valid",
                label="API Credentials",
                description="Coinbase API credentials validated",
                completed=self._check_credentials_valid(mode),
                required=not is_demo,
                skippable=is_demo,
            ),
            ChecklistItem(
                id="connection_ok",
                label="Connection",
                description="Connected to data source",
                completed=self._check_connection(state, is_demo),
                required=True,
            ),
            ChecklistItem(
                id="risk_reviewed",
                label="Risk Settings",
                description="Risk limits reviewed",
                completed=self._check_risk_reviewed(),
                required=False,  # Nice to have, not blocking
            ),
        ]

        return OnboardingStatus(items=items, mode=mode)

    def _check_credentials_valid(self, mode: str) -> bool:
        """Check if credentials are validated for the given mode.

        Args:
            mode: Trading mode to check.

        Returns:
            True if credentials are valid or not needed.
        """
        if mode == "demo":
            return True

        try:
            prefs = self._preferences._preferences
            if prefs is None or not prefs.credential_validation_modes:
                return False

            # Check if this mode is validated
            return prefs.credential_validation_modes.get(mode, False)
        except Exception:
            return False

    def _check_connection(self, state: TuiState | None, is_demo: bool) -> bool:
        """Check if connection is established.

        Args:
            state: Current TUI state.
            is_demo: Whether running in demo mode.

        Returns:
            True if connected or in demo mode.
        """
        if is_demo:
            return True

        if state is None:
            return False

        conn_status = state.system_data.connection_status
        return conn_status in ("CONNECTED", "READY", "OK")

    def _check_risk_reviewed(self) -> bool:
        """Check if risk settings have been reviewed.

        Returns:
            True if user has acknowledged risk settings.
        """
        # For now, we'll consider this done if they've used the app before
        # (have a saved mode preference)
        try:
            prefs = self._preferences._preferences
            if prefs is None:
                return False
            return prefs.mode is not None
        except Exception:
            return False

    def mark_risk_reviewed(self) -> None:
        """Mark risk settings as reviewed."""
        # This is implicitly done by having used the app
        pass


# Global singleton
_onboarding_service: OnboardingService | None = None


def get_onboarding_service() -> OnboardingService:
    """Get or create the global onboarding service."""
    global _onboarding_service
    if _onboarding_service is None:
        _onboarding_service = OnboardingService()
    return _onboarding_service


def clear_onboarding_service() -> None:
    """Clear the global onboarding service (for testing)."""
    global _onboarding_service
    _onboarding_service = None


__all__ = [
    "ChecklistItem",
    "OnboardingService",
    "OnboardingStatus",
    "clear_onboarding_service",
    "get_onboarding_service",
]
