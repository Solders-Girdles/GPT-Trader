"""Guard rail manager for enforcing trading safety policies."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass
class OrderCheckResult:
    allowed: bool
    guard: str | None = None
    reason: str | None = None


class GuardRailManager:
    def __init__(
        self,
        max_trade_value: Decimal | None = None,
        symbol_position_caps: dict[str, Decimal] | None = None,
        daily_loss_limit: Decimal | None = None,
        error_threshold: int = 5,
        error_cooldown_seconds: float = 300.0,
    ) -> None:
        self._guards: dict[str, bool] = {}
        self._listeners: list[Callable[[str, bool], None]] = []
        self._max_trade_value = max_trade_value or Decimal("0")
        self._symbol_position_caps = symbol_position_caps or {}
        self._daily_loss_limit = daily_loss_limit or Decimal("0")
        self._error_threshold = max(error_threshold, 0)
        self._error_cooldown_seconds = max(error_cooldown_seconds, 0.0)

        # Daily P&L tracking
        self._daily_realized_pnl = Decimal("0")
        self._pnl_tracking_date: date | None = None

        # Error streak tracking
        self._error_streak: int = 0
        self._last_error_timestamp: float | None = None

    def register_listener(self, listener: Callable[[str, bool], None]) -> None:
        self._listeners.append(listener)

    def _set_guard_state(self, name: str, active: bool) -> None:
        previous = self._guards.get(name)
        if previous == active:
            return
        self._guards[name] = active
        for listener in self._listeners:
            listener(name, active)

    def is_guard_active(self, name: str) -> bool:
        return bool(self._guards.get(name, False))

    def check_order(self, context: dict[str, Any]) -> OrderCheckResult:
        if self.is_guard_active("dry_run"):
            return OrderCheckResult(False, guard="dry_run", reason="dry_run_active")

        # Check max trade value limit
        if self._max_trade_value > 0:
            mark = context.get("mark")
            order_kwargs = context.get("order_kwargs", {})
            quantity = order_kwargs.get("quantity")

            if mark and quantity:
                try:
                    notional = Decimal(str(mark)) * Decimal(str(quantity))
                    if notional > self._max_trade_value:
                        return OrderCheckResult(
                            False,
                            guard="max_trade_value",
                            reason=f"order_value_{notional}_exceeds_limit_{self._max_trade_value}",
                        )
                except Exception:
                    # If we can't compute notional, allow the order (fail-open for safety)
                    pass

        # Check symbol position caps
        symbol = context.get("symbol")
        if symbol and symbol in self._symbol_position_caps:
            cap = self._symbol_position_caps[symbol]
            order_kwargs = context.get("order_kwargs", {})
            quantity = order_kwargs.get("quantity")

            if quantity:
                try:
                    order_qty = abs(Decimal(str(quantity)))
                    if order_qty > cap:
                        return OrderCheckResult(
                            False,
                            guard="position_limit",
                            reason=f"symbol_{symbol}_quantity_{order_qty}_exceeds_cap_{cap}",
                        )
                except Exception:
                    # If we can't compute quantity, allow the order (fail-open for safety)
                    pass

        return OrderCheckResult(True)

    def check_cycle(self, context: dict[str, Any]) -> None:
        """Check cycle-level guards (e.g., daily loss limit).

        Args:
            context: Cycle context with positions, balances, etc.
        """
        # Reset daily tracking if new day
        today = date.today()
        if self._pnl_tracking_date != today:
            logger.info("New trading day detected, resetting daily P&L tracking")
            self._daily_realized_pnl = Decimal("0")
            self._pnl_tracking_date = today
            # Clear daily_loss guard if it was active
            if self.is_guard_active("daily_loss"):
                self._set_guard_state("daily_loss", False)

        # Check daily loss limit
        if self._daily_loss_limit > 0:
            # Negative P&L means losses
            daily_loss = abs(min(self._daily_realized_pnl, Decimal("0")))

            if daily_loss >= self._daily_loss_limit:
                if not self.is_guard_active("daily_loss"):
                    logger.warning(
                        "Daily loss limit reached: $%s >= $%s - activating reduce-only guard",
                        daily_loss,
                        self._daily_loss_limit,
                    )
                    self._set_guard_state("daily_loss", True)

        # Circuit breaker cooldown/reset
        if self._error_streak > 0 and self._last_error_timestamp:
            if time.time() - self._last_error_timestamp >= self._error_cooldown_seconds:
                logger.info("Error streak cooldown elapsed; resetting streak and circuit breaker")
                self._error_streak = 0
                self._last_error_timestamp = None
                if self.is_guard_active("circuit_breaker"):
                    self._set_guard_state("circuit_breaker", False)

    def set_dry_run(self, enabled: bool) -> None:
        self._set_guard_state("dry_run", enabled)

    def update_limits(
        self,
        max_trade_value: Decimal | None = None,
        symbol_position_caps: dict[str, Decimal] | None = None,
        daily_loss_limit: Decimal | None = None,
    ) -> None:
        """Update order cap limits dynamically."""
        if max_trade_value is not None:
            self._max_trade_value = max_trade_value
        if symbol_position_caps is not None:
            self._symbol_position_caps = symbol_position_caps
        if daily_loss_limit is not None:
            self._daily_loss_limit = daily_loss_limit

    def record_realized_pnl(self, pnl: Decimal) -> None:
        """Record realized P&L for daily loss tracking.

        Args:
            pnl: Realized P&L amount (positive = profit, negative = loss)
        """
        today = date.today()
        if self._pnl_tracking_date != today:
            # New day, reset tracking
            self._daily_realized_pnl = Decimal("0")
            self._pnl_tracking_date = today

        self._daily_realized_pnl += pnl
        logger.debug("Recorded realized P&L: %s, daily total: %s", pnl, self._daily_realized_pnl)

    def get_daily_pnl(self) -> Decimal:
        """Get current daily realized P&L.

        Returns:
            Current daily P&L (positive = profit, negative = loss)
        """
        today = date.today()
        if self._pnl_tracking_date != today:
            return Decimal("0")
        return self._daily_realized_pnl

    def get_daily_loss(self) -> Decimal:
        """Get current daily loss amount (always positive).

        Returns:
            Current daily loss as positive number (0 if profitable)
        """
        return abs(min(self.get_daily_pnl(), Decimal("0")))

    def record_error(self, reason: str, *, critical: bool = True) -> tuple[int, bool]:
        """Record an error event and update circuit breaker state.

        Args:
            reason: Human-readable reason for logging/metrics
            critical: If False, does not contribute to streak

        Returns:
            Tuple of (current error streak, guard_triggered_flag)
        """
        if not critical:
            return self._error_streak, False

        self._error_streak += 1
        self._last_error_timestamp = time.time()
        guard_triggered = False

        if self._error_threshold > 0 and self._error_streak >= self._error_threshold:
            if not self.is_guard_active("circuit_breaker"):
                logger.warning(
                    "Circuit breaker threshold reached (%s errors). Reason: %s",
                    self._error_streak,
                    reason,
                )
                self._set_guard_state("circuit_breaker", True)
                guard_triggered = True

        return self._error_streak, guard_triggered

    def record_success(self) -> None:
        """Reset error streak after successful operation."""
        if self._error_streak > 0:
            logger.debug("Resetting error streak from %s to 0", self._error_streak)
        self._error_streak = 0
        self._last_error_timestamp = None
        if self.is_guard_active("circuit_breaker"):
            self._set_guard_state("circuit_breaker", False)

    def get_error_streak(self) -> int:
        """Return current error streak."""
        return self._error_streak

    def snapshot(self) -> dict[str, bool]:
        return dict(self._guards)
