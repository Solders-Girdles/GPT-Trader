"""
State tracking and mode management for risk system.

Handles reduce-only mode, daily tracking resets, and state listeners.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from bot_v2.orchestration.configuration import RiskConfig
from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="live_trade_risk")


@dataclass
class RiskRuntimeState:
    reduce_only_mode: bool = False
    last_reduce_only_reason: str | None = None
    last_reduce_only_at: datetime | None = None


class StateManager:
    """Manages risk system state and mode transitions."""

    def __init__(
        self,
        config: RiskConfig,
        event_store: EventStore,
        now_provider: Callable[[], datetime] | None = None,
    ):
        """
        Initialize state manager.

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

        timestamp = self._now_provider() if enabled else None
        new_state = RiskRuntimeState(
            reduce_only_mode=enabled,
            last_reduce_only_reason=(reason or "unspecified") if enabled else None,
            last_reduce_only_at=timestamp,
        )
        self._state = new_state

        # Mirror change onto config so legacy access patterns remain valid
        try:
            self.config.reduce_only_mode = enabled
        except Exception as exc:
            logger.debug("Failed to mirror reduce-only state onto config: %s", exc, exc_info=True)

        emit_metric(
            self.event_store,
            "risk_engine",
            {
                "event_type": "reduce_only_mode_changed",
                "enabled": enabled,
                "reason": reason or "unspecified",
                "timestamp": (timestamp.isoformat() if timestamp else None),
            },
            logger=logger,
        )

        if self._state_listener:
            try:
                self._state_listener(new_state)
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
