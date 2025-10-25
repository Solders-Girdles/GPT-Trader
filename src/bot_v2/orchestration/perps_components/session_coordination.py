"""Session coordination functionality separated from perps_bot.py.

This module contains session coordination logic that was previously
embedded in the large perps_bot.py file. It provides:

- Trading session management and validation
- Time-based trading windows
- Session state tracking
- Integration with bot lifecycle
- Session-based trading controls
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.config_controller import ConfigController
    from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
    from bot_v2.orchestration.session_guard import TradingSessionGuard

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="perps_session_coordination")


class SessionCoordinationService:
    """Service responsible for session coordination and trading window management.

    This service consolidates session-related logic that was previously
    embedded in the PerpsBot class, providing focused responsibility
    for trading session management and validation.
    """

    def __init__(
        self,
        config_controller: ConfigController,
        bot_state: PerpsBotRuntimeState,
        session_guard: TradingSessionGuard,
    ) -> None:
        """Initialize session coordination service.

        Args:
            config_controller: Configuration management controller
            bot_state: Runtime state instance for the bot
            session_guard: Trading session guard for time windows
        """
        self.config_controller = config_controller
        self.bot_state = bot_state
        self.session_guard = session_guard

    def validate_trading_session(self) -> dict[str, Any]:
        """Validate if trading is currently allowed."""
        session_status = self.session_guard.get_session_status()

        return {
            "trading_allowed": session_status.is_trading_allowed,
            "in_trading_window": session_status.in_window,
            "window_start": (
                session_status.window_start.isoformat() if session_status.window_start else None
            ),
            "window_end": (
                session_status.window_end.isoformat() if session_status.window_end else None
            ),
            "trading_days": session_status.trading_days,
            "session_guard_active": True,
            "validation_message": self._get_session_message(session_status),
        }

    def _get_session_message(self, session_status: Any) -> str:
        """Get user-friendly session status message."""
        if session_status.is_trading_allowed:
            return "Trading allowed"
        elif session_status.in_window:
            return "Outside trading hours"
        elif (
            session_status.trading_days
            and datetime.now().strftime("%A") not in session_status.trading_days
        ):
            return "Not a trading day"
        else:
            return "Trading session guard active"

    def enforce_session_restrictions(self, operation: str, action_func: Any, *args: Any) -> Any:
        """Enforce session restrictions before executing actions."""
        if not self.session_guard.is_trading_allowed():
            session_status = self.session_guard.get_session_status()
            logger.warning(
                f"Operation blocked: {operation}",
                operation="session_block",
                blocked_operation=operation,
                reason=self._get_session_message(session_status),
                in_window=session_status.in_window,
            )
            raise RuntimeError(
                f"Operation not allowed: {self._get_session_message(session_status)}"
            )

        return action_func(*args)

    def get_session_health_status(self) -> dict[str, Any]:
        """Get health status of session coordination."""
        session_status = self.session_guard.get_session_status()

        return {
            "session_guard_healthy": True,  # Always healthy unless exceptions
            "trading_allowed": session_status.is_trading_allowed,
            "in_trading_window": session_status.in_window,
            "trading_days_compliant": not session_status.trading_days
            or datetime.now().strftime("%A") in session_status.trading_days,
            "next_window_change": session_status.next_change,
            "validation_message": self._get_session_message(session_status),
        }


__all__ = [
    "SessionCoordinationService",
]
