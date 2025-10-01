"""
State tracking and mode management for risk system.

Handles reduce-only mode, daily tracking resets, and state listeners.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.persistence.event_store import EventStore

logger = logging.getLogger(__name__)


@dataclass
class RiskRuntimeState:
    reduce_only_mode: bool = False
    last_reduce_only_reason: str | None = None
    last_reduce_only_at: datetime | None = None


class RiskStateManager:
    """Manages risk system state and mode transitions.

    Note: Renamed from StateManager to avoid confusion with bot_v2.state.state_manager.StateManager.
    """

    def __init__(
        self,
        config: RiskConfig,
        event_store: EventStore,
        now_provider: Callable[[], datetime] | None = None,
    ):
        """
        Initialize risk state manager.

        Args:
            config: Risk configuration
            event_store: Event store for state change persistence
            now_provider: Time provider for testability
        """
        self.config = config
        self.event_store = event_store
        self._now_provider = now_provider or (lambda: datetime.utcnow())

        # Runtime state
        self._state = RiskRuntimeState(reduce_only_mode=bool(config.reduce_only_mode))
        self._state_listener: Callable[[RiskRuntimeState], None] | None = None

        # Daily tracking
        self.daily_pnl = Decimal("0")
        self.start_of_day_equity = Decimal("0")

    def is_reduce_only_mode(self) -> bool:
        """Check if reduce-only mode is active."""
        return self._state.reduce_only_mode or bool(getattr(self.config, "reduce_only_mode", False))

    def set_reduce_only_mode(self, enabled: bool, reason: str = "") -> None:
        """
        Toggle reduce-only mode without mutating the shared config object.

        Args:
            enabled: Whether to enable reduce-only mode
            reason: Reason for the mode change
        """
        if (
            self._state.reduce_only_mode == enabled
            and bool(getattr(self.config, "reduce_only_mode", False)) == enabled
        ):
            return

        self._state.reduce_only_mode = enabled
        if enabled:
            self._state.last_reduce_only_reason = reason or "unspecified"
            self._state.last_reduce_only_at = self._now_provider()
        else:
            self._state.last_reduce_only_reason = None
            self._state.last_reduce_only_at = None

        # Mirror change onto config so legacy access patterns remain valid
        try:
            self.config.reduce_only_mode = enabled
        except Exception as exc:
            logger.debug("Failed to mirror reduce-only state onto config: %s", exc, exc_info=True)

        try:
            self.event_store.append_metric(
                bot_id="risk_engine",
                metrics={
                    "event_type": "reduce_only_mode_changed",
                    "enabled": enabled,
                    "reason": reason or "unspecified",
                    "timestamp": (
                        self._state.last_reduce_only_at.isoformat()
                        if self._state.last_reduce_only_at
                        else None
                    ),
                },
            )
        except Exception as exc:
            logger.warning("Failed to persist reduce-only mode change: %s", exc)

        if self._state_listener:
            try:
                self._state_listener(self._state)
            except Exception:
                logger.exception("Reduce-only state listener failed")

    def set_state_listener(self, listener: Callable[[RiskRuntimeState], None] | None) -> None:
        """Register a listener for state changes."""
        self._state_listener = listener

    def reset_daily_tracking(self, current_equity: Decimal) -> None:
        """Reset daily tracking at start of new day."""
        self.daily_pnl = Decimal("0")
        self.start_of_day_equity = current_equity
        logger.info(f"Reset daily tracking - Start of day equity: {current_equity}")
