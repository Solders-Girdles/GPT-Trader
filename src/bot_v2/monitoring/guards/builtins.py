"""Built-in guard implementations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from decimal import Decimal, DecimalException
from typing import Any

from .base import GuardConfig, RuntimeGuard


class DailyLossGuard(RuntimeGuard):
    """Monitor daily loss limits."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.daily_pnl: Decimal = Decimal("0")
        self.last_reset = datetime.now().date()

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = Decimal("0")
            self.last_reset = current_date

        try:
            pnl = Decimal(str(context.get("pnl", 0)))
        except (TypeError, ValueError, DecimalException):
            return False, ""
        self.daily_pnl += pnl

        threshold = Decimal(str(abs(self.config.threshold)))
        if threshold == Decimal("0"):
            return False, ""

        if self.daily_pnl < -threshold:
            loss_amount = abs(self.daily_pnl)
            message = (
                f"Daily loss limit breached: ${loss_amount:.2f} "
                f"(limit: ${self.config.threshold:.2f})"
            )
            return True, message

        warning_threshold = threshold * Decimal("0.5")
        if self.daily_pnl <= -warning_threshold and self.status == self.status.HEALTHY:
            self.status = self.status.WARNING
        return False, ""


class StaleMarkGuard(RuntimeGuard):
    """Monitor for stale market data."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.last_marks: dict[str, datetime] = {}

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        symbol = context.get("symbol")
        mark_time = context.get("mark_timestamp")

        if not isinstance(symbol, str) or mark_time is None:
            return False, ""

        if isinstance(mark_time, str):
            try:
                mark_time = datetime.fromisoformat(mark_time)
            except ValueError:
                return False, ""
        elif isinstance(mark_time, (int, float)):
            mark_time = datetime.fromtimestamp(mark_time)
        elif not isinstance(mark_time, datetime):
            return False, ""

        self.last_marks[symbol] = mark_time

        age_seconds = (datetime.now() - mark_time).total_seconds()
        if age_seconds > self.config.threshold:
            message = (
                f"Stale marks detected for {symbol}: "
                f"{age_seconds:.1f}s old (limit: {self.config.threshold}s)"
            )
            return True, message
        return False, ""


class ErrorRateGuard(RuntimeGuard):
    """Monitor error rates."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.error_times: list[datetime] = []

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        if context.get("error"):
            self.error_times.append(datetime.now())

        cutoff = datetime.now() - timedelta(seconds=self.config.window_seconds)
        self.error_times = [t for t in self.error_times if t > cutoff]

        error_count = len(self.error_times)
        if error_count > self.config.threshold:
            message = (
                f"High error rate: {error_count} errors in "
                f"{self.config.window_seconds}s (limit: {int(self.config.threshold)})"
            )
            return True, message
        return False, ""


class PositionStuckGuard(RuntimeGuard):
    """Monitor for positions that aren't being managed."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.position_times: dict[str, datetime] = {}

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        positions = context.get("positions", {})
        if not isinstance(positions, Mapping):
            return False, ""

        for symbol, position in positions.items():
            if not isinstance(symbol, str) or not isinstance(position, Mapping):
                continue
            size_value = position.get(
                "quantity", position.get("size", position.get("contracts", 0))
            )
            try:
                size = float(size_value)
            except (TypeError, ValueError):
                continue
            if size != 0.0:
                self.position_times.setdefault(symbol, datetime.now())
            else:
                self.position_times.pop(symbol, None)

        stuck_positions: list[tuple[str, float]] = []
        for symbol, open_time in list(self.position_times.items()):
            age_seconds = (datetime.now() - open_time).total_seconds()
            if age_seconds > self.config.threshold:
                stuck_positions.append((symbol, age_seconds))

        if stuck_positions:
            details = ", ".join([f"{sym}: {age:.0f}s" for sym, age in stuck_positions])
            message = f"Stuck positions detected: {details}"
            return True, message
        return False, ""


class DrawdownGuard(RuntimeGuard):
    """Monitor maximum drawdown."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.peak_equity: Decimal = Decimal("0")
        self.current_drawdown: Decimal = Decimal("0")

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        try:
            equity = Decimal(str(context.get("equity", 0)))
        except (TypeError, ValueError, DecimalException):
            return False, ""

        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > Decimal("0"):
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity * Decimal(100)
            threshold = Decimal(str(self.config.threshold))

            if self.current_drawdown > threshold:
                message = (
                    f"Maximum drawdown breached: {self.current_drawdown:.2f}% "
                    f"(limit: {self.config.threshold:.2f}%)"
                )
                return True, message
        return False, ""


__all__ = [
    "DailyLossGuard",
    "StaleMarkGuard",
    "ErrorRateGuard",
    "PositionStuckGuard",
    "DrawdownGuard",
]
