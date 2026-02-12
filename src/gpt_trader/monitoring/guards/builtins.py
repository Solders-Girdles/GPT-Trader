"""Built-in guard implementations."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from decimal import Decimal, DecimalException
from typing import Any

from gpt_trader.utilities.datetime_helpers import normalize_to_utc
from gpt_trader.utilities.time_provider import TimeProvider

from .base import GuardConfig, GuardStatus, RuntimeGuard


class DailyLossGuard(RuntimeGuard):
    """Monitor daily loss limits."""

    def __init__(self, config: GuardConfig, *, time_provider: TimeProvider | None = None) -> None:
        super().__init__(config, time_provider=time_provider)
        self.daily_pnl: Decimal = Decimal("0")
        self.last_reset = self._now().date()

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        current_date = self._now().date()
        if current_date > self.last_reset:
            self.daily_pnl = Decimal("0")
            self.last_reset = current_date
            self.status = GuardStatus.HEALTHY

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

    def __init__(self, config: GuardConfig, *, time_provider: TimeProvider | None = None) -> None:
        super().__init__(config, time_provider=time_provider)
        self.last_marks: dict[str, datetime] = {}

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        symbol = context.get("symbol")
        mark_time = context.get("mark_timestamp")

        if not isinstance(symbol, str) or mark_time is None:
            return False, ""

        if isinstance(mark_time, str):
            try:
                mark_time = normalize_to_utc(datetime.fromisoformat(mark_time))
            except ValueError:
                return False, ""
        elif isinstance(mark_time, (int, float)):
            mark_time = datetime.fromtimestamp(mark_time, UTC)
        elif isinstance(mark_time, datetime):
            mark_time = normalize_to_utc(mark_time)
        else:
            return False, ""

        self.last_marks[symbol] = mark_time

        age_seconds = (self._now() - mark_time).total_seconds()
        if age_seconds > self.config.threshold:
            message = (
                f"Stale marks detected for {symbol}: "
                f"{age_seconds:.1f}s old (limit: {self.config.threshold}s)"
            )
            return True, message
        return False, ""


class ErrorRateGuard(RuntimeGuard):
    """Monitor error rates."""

    def __init__(self, config: GuardConfig, *, time_provider: TimeProvider | None = None) -> None:
        super().__init__(config, time_provider=time_provider)
        self.error_times: list[datetime] = []

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        if context.get("error"):
            self.error_times.append(self._now())

        cutoff = self._now() - timedelta(seconds=self.config.window_seconds)
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

    def __init__(self, config: GuardConfig, *, time_provider: TimeProvider | None = None) -> None:
        super().__init__(config, time_provider=time_provider)
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
                self.position_times.setdefault(symbol, self._now())
            else:
                self.position_times.pop(symbol, None)

        stuck_positions: list[tuple[str, float]] = []
        for symbol, open_time in list(self.position_times.items()):
            age_seconds = (self._now() - open_time).total_seconds()
            if age_seconds > self.config.threshold:
                stuck_positions.append((symbol, age_seconds))

        if stuck_positions:
            details = ", ".join([f"{sym}: {age:.0f}s" for sym, age in stuck_positions])
            message = f"Stuck positions detected: {details}"
            return True, message
        return False, ""


class DrawdownGuard(RuntimeGuard):
    """Monitor maximum drawdown."""

    def __init__(self, config: GuardConfig, *, time_provider: TimeProvider | None = None) -> None:
        super().__init__(config, time_provider=time_provider)
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


class LauncherStarvationGuard(RuntimeGuard):
    """Monitor repeated launcher no-candidate streaks."""

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        streak_raw = context.get("no_candidate_streak")
        if streak_raw is None:
            return False, ""

        try:
            streak_value = int(streak_raw)
        except (TypeError, ValueError):
            return False, ""

        threshold_override = context.get("threshold_override")
        if threshold_override is None:
            threshold_override = context.get("threshold")
        threshold_decimal = self._coerce_decimal(threshold_override)
        if threshold_decimal is None:
            threshold_decimal = self._coerce_decimal(self.config.threshold)
        if threshold_decimal is None:
            threshold_value = 0
        else:
            threshold_value = int(threshold_decimal)

        if streak_value <= threshold_value:
            return False, ""

        opportunity_id = context.get("opportunity_id")
        message = (
            f"Launcher reported {streak_value} consecutive no-candidate cycles "
            f"(threshold: {threshold_value})."
        )
        if isinstance(opportunity_id, str) and opportunity_id:
            message += f" Last opportunity: {opportunity_id}."

        return True, message


__all__ = [
    "DailyLossGuard",
    "StaleMarkGuard",
    "ErrorRateGuard",
    "PositionStuckGuard",
    "DrawdownGuard",
    "LauncherStarvationGuard",
]
