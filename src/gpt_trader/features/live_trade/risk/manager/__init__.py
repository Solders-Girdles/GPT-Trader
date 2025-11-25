import time
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


class ValidationError(Exception):
    pass


@dataclass
class VolatilityCheckOutcome:
    """Result of a volatility circuit breaker check."""

    triggered: bool = False
    symbol: str = ""
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "triggered": self.triggered,
            "symbol": self.symbol,
            "reason": self.reason,
        }


class LiveRiskManager:
    def __init__(self, config: Any = None, event_store: Any = None) -> None:
        self.config = config
        self.event_store = event_store
        self.positions: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        self.last_mark_update: dict[str, float] = {}
        self._reduce_only_mode: bool = False
        self._reduce_only_reason: str = ""
        self._daily_pnl_triggered: bool = False
        self._risk_metrics: list[dict[str, Any]] = []
        self._start_of_day_equity: Decimal | None = None

    def check_order(self, order: Any) -> bool:
        return True

    def update_position(self, position: Any) -> None:
        pass

    def check_liquidation_buffer(self, symbol: str, position: Any, equity: Decimal) -> bool:
        if not self.config:
            return False

        try:
            # Handle object or dict
            if isinstance(position, dict):
                liq_price = position.get("liquidation_price")
                mark_price = position.get("mark") or position.get("mark_price")
            else:
                liq_price = getattr(position, "liquidation_price", None)
                mark_price = getattr(position, "mark_price", None) or getattr(
                    position, "mark", None
                )

            if not liq_price or not mark_price:
                return False

            mark_price = Decimal(str(mark_price))
            liq_price = Decimal(str(liq_price))

            if mark_price == 0:
                return False

            buffer_pct = abs(mark_price - liq_price) / mark_price

            if buffer_pct < self.config.min_liquidation_buffer_pct:
                self.positions[symbol]["reduce_only"] = True
                return True

        except (AttributeError, TypeError, ZeroDivisionError, ValueError):
            pass

        return False

    def pre_trade_validate(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
        product: Any,
        equity: Decimal,
        current_positions: dict[str, Any],
    ) -> None:
        if self.config:
            notional = quantity * price
            if equity > 0:
                leverage = notional / equity
                if leverage > self.config.max_leverage:
                    raise ValidationError(
                        f"Leverage {leverage} exceeds max {self.config.max_leverage}"
                    )

    def track_daily_pnl(
        self, equity: Decimal, positions_pnl: dict[str, dict[str, Decimal]]
    ) -> bool:
        """
        Track daily PnL and check if loss limit is breached.

        Returns:
            True if daily loss limit was triggered, False otherwise.
        """
        if self._start_of_day_equity is None:
            self._start_of_day_equity = equity
            return False

        if not self.config:
            return False

        daily_loss_limit = getattr(self.config, "daily_loss_limit_pct", None)
        if not daily_loss_limit:
            return False

        daily_pnl = equity - self._start_of_day_equity
        if self._start_of_day_equity > 0:
            loss_pct = -daily_pnl / self._start_of_day_equity
            if loss_pct > Decimal(str(daily_loss_limit)):
                self._daily_pnl_triggered = True
                self.set_reduce_only_mode(True, reason="daily_loss_limit_breached")
                return True
        return False

    def check_mark_staleness(self, symbol: str) -> bool:
        """
        Check if mark price for symbol is stale.

        Returns:
            True if mark is stale (or missing), False if fresh.
        """
        last_update = self.last_mark_update.get(symbol)
        if last_update is None:
            return True

        staleness_threshold = 120.0  # seconds
        if self.config and hasattr(self.config, "mark_staleness_threshold"):
            staleness_threshold = float(self.config.mark_staleness_threshold)

        return (time.time() - last_update) > staleness_threshold

    def append_risk_metrics(
        self, equity: Decimal, positions_dict: dict[str, dict[str, Decimal]]
    ) -> None:
        """Append current risk metrics for monitoring."""
        metrics = {
            "timestamp": time.time(),
            "equity": str(equity),
            "positions": {
                k: {kk: str(vv) for kk, vv in v.items()} for k, v in positions_dict.items()
            },
            "reduce_only_mode": self._reduce_only_mode,
        }
        self._risk_metrics.append(metrics)
        # Keep only last 100 metrics
        if len(self._risk_metrics) > 100:
            self._risk_metrics = self._risk_metrics[-100:]

    def check_correlation_risk(self, positions_dict: dict[str, dict[str, Decimal]]) -> None:
        """
        Check correlation risk across positions.
        Currently a no-op placeholder for future implementation.
        """
        pass

    def check_volatility_circuit_breaker(
        self, symbol: str, closes: list[Decimal]
    ) -> VolatilityCheckOutcome:
        """
        Check if volatility exceeds threshold for the symbol.

        Returns:
            VolatilityCheckOutcome with triggered=True if breaker should trip.
        """
        if not closes or len(closes) < 5:
            return VolatilityCheckOutcome(triggered=False, symbol=symbol)

        if not self.config:
            return VolatilityCheckOutcome(triggered=False, symbol=symbol)

        threshold = getattr(self.config, "volatility_threshold_pct", None)
        if not threshold:
            return VolatilityCheckOutcome(triggered=False, symbol=symbol)

        # Calculate simple volatility as max deviation from mean
        avg_close = sum(closes) / len(closes)
        if avg_close == 0:
            return VolatilityCheckOutcome(triggered=False, symbol=symbol)

        max_deviation = max(abs(c - avg_close) for c in closes)
        volatility_pct = max_deviation / avg_close

        if volatility_pct > Decimal(str(threshold)):
            self.set_reduce_only_mode(True, reason=f"volatility_breaker_{symbol}")
            return VolatilityCheckOutcome(
                triggered=True,
                symbol=symbol,
                reason=f"Volatility {volatility_pct:.2%} exceeds threshold {threshold}",
            )

        return VolatilityCheckOutcome(triggered=False, symbol=symbol)

    def set_reduce_only_mode(self, value: bool, reason: str = "") -> None:
        """Set the reduce-only trading mode."""
        self._reduce_only_mode = value
        self._reduce_only_reason = reason

    def is_reduce_only_mode(self) -> bool:
        """Check if reduce-only mode is active."""
        return self._reduce_only_mode

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking state (call at start of each trading day)."""
        self._start_of_day_equity = None
        self._daily_pnl_triggered = False
        self._reduce_only_mode = False
        self._reduce_only_reason = ""
