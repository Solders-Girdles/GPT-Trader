"""
Live risk management for perpetuals trading.

This module provides the LiveRiskManager class which enforces:
- Maximum leverage limits (global and per-symbol)
- Day/night leverage caps with configurable time windows
- Daily loss limits with automatic reduce-only mode
- Position exposure limits
- Volatility circuit breakers
- Mark price staleness detection
- Liquidation buffer monitoring
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpt_trader.orchestration.configuration.risk import RiskConfig
    from gpt_trader.persistence.event_store import EventStore


class ValidationError(Exception):
    """Raised when a trade fails risk validation checks."""

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
    """Manages risk controls for live perpetuals trading.

    Enforces leverage limits, daily loss limits, exposure caps, and circuit breakers.
    Integrates with EventStore for metrics/alerting.
    """

    def __init__(
        self,
        config: RiskConfig | None = None,
        event_store: EventStore | None = None,
        state_file: str | None = "var/data/risk_state.json",
    ) -> None:
        """
        Initialize the risk manager.

        Args:
            config: Risk configuration with leverage limits, loss limits, etc.
            event_store: Event store for recording risk metrics.
            state_file: Path to JSON file for persisting daily risk state.
        """
        self.config = config
        self.event_store = event_store
        self.state_file = state_file
        self.positions: defaultdict[str, dict[str, Any]] = defaultdict(dict)
        self.last_mark_update: dict[str, float] = {}
        self._reduce_only_mode: bool = False
        self._reduce_only_reason: str = ""
        self._daily_pnl_triggered: bool = False
        self._risk_metrics: list[dict[str, Any]] = []
        self._start_of_day_equity: Decimal | None = None
        # Allows time mocking for tests; defaults to real datetime.utcnow
        self._now_provider: Callable[[], datetime] = datetime.utcnow

        # Load persisted state if available
        if self.state_file:
            self._load_state()

    def _load_state(self) -> None:
        """Load risk state from disk."""
        import json
        import os

        if not self.state_file or not os.path.exists(self.state_file):
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            # Check if state is from today (UTC)
            saved_date = state.get("date")
            current_date = self._now_provider().strftime("%Y-%m-%d")

            if saved_date == current_date:
                if state.get("start_of_day_equity"):
                    self._start_of_day_equity = Decimal(str(state["start_of_day_equity"]))
                self._daily_pnl_triggered = state.get("daily_pnl_triggered", False)
                self._reduce_only_mode = state.get("reduce_only_mode", False)
                self._reduce_only_reason = state.get("reduce_only_reason", "")
                # Restore reduce-only flags for positions if needed
                # (For now we just restore global flags)
            else:
                # New day, reset state but keep file for history?
                # Actually, reset_daily_tracking will handle logic, but here we just don't load stale state
                pass

        except Exception as e:
            # Log error but don't crash
            print(f"Failed to load risk state: {e}")

    def _save_state(self) -> None:
        """Save risk state to disk."""
        import json

        if not self.state_file:
            return

        try:
            # Ensure directory exists
            from pathlib import Path

            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)

            state = {
                "date": self._now_provider().strftime("%Y-%m-%d"),
                "start_of_day_equity": (
                    str(self._start_of_day_equity) if self._start_of_day_equity else None
                ),
                "daily_pnl_triggered": self._daily_pnl_triggered,
                "reduce_only_mode": self._reduce_only_mode,
                "reduce_only_reason": self._reduce_only_reason,
                "updated_at": time.time(),
            }

            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"Failed to save risk state: {e}")

    def check_order(self, order: Any) -> bool:
        """Check if an order is allowed by risk rules.

        This is a quick check for order-level validation.
        For comprehensive pre-trade validation, use pre_trade_validate() instead.

        Returns:
            True if order is allowed, False if blocked by risk rules.
        """
        # Check if daily loss limit was triggered
        if self._daily_pnl_triggered:
            return False

        # Check reduce-only mode
        if self._reduce_only_mode:
            # In reduce-only mode, only allow orders that reduce position
            reduce_only = False
            if hasattr(order, "reduce_only"):
                reduce_only = order.reduce_only
            elif isinstance(order, dict):
                reduce_only = order.get("reduce_only", False)

            if not reduce_only:
                return False

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
        if not self.config:
            return

        notional = quantity * price

        if equity <= 0:
            return

        leverage = notional / equity

        # 1. Global max leverage check
        if leverage > self.config.max_leverage:
            raise ValidationError(f"Leverage {leverage} exceeds max {self.config.max_leverage}")

        # 2. Day/night leverage cap check (per-symbol)
        is_daytime = self._is_daytime()
        symbol_leverage_cap = self._get_symbol_leverage_cap(symbol, is_daytime)
        if symbol_leverage_cap is not None and leverage > symbol_leverage_cap:
            time_period = "day" if is_daytime else "night"
            raise ValidationError(
                f"Leverage {leverage:.1f}x exceeds {symbol} cap of {symbol_leverage_cap}x ({time_period})"
            )

        # 3. Total exposure validation
        self._validate_total_exposure(symbol, notional, equity, current_positions)

        # 4. MMR projection check (liquidation buffer)
        if getattr(self.config, "enable_pre_trade_liq_projection", False):
            self._validate_mmr_projection(symbol, notional, equity, is_daytime)

    def _is_daytime(self) -> bool:
        """Check if current time is within daytime window."""
        now = self._now_provider()
        current_time_str = now.strftime("%H:%M")

        daytime_start = getattr(self.config, "daytime_start_utc", "09:00")
        daytime_end = getattr(self.config, "daytime_end_utc", "17:00")

        return daytime_start <= current_time_str < daytime_end

    def _get_symbol_leverage_cap(self, symbol: str, is_daytime: bool) -> int | None:
        """Get the leverage cap for a symbol based on time of day."""
        if is_daytime:
            caps = getattr(self.config, "day_leverage_max_per_symbol", {})
        else:
            caps = getattr(self.config, "night_leverage_max_per_symbol", {})

        return caps.get(symbol)

    def _validate_total_exposure(
        self,
        symbol: str,
        new_notional: Decimal,
        equity: Decimal,
        current_positions: dict[str, Any],
    ) -> None:
        """Validate that total exposure doesn't exceed cap."""
        max_exposure_pct = getattr(self.config, "max_exposure_pct", 1.0)

        # Calculate current total exposure
        current_exposure = Decimal("0")
        for pos_symbol, pos_data in current_positions.items():
            if isinstance(pos_data, dict):
                quantity = Decimal(str(pos_data.get("quantity", 0)))
                mark = Decimal(str(pos_data.get("mark", pos_data.get("price", 0))))
                current_exposure += abs(quantity * mark)

        # Add the new position notional
        total_exposure = current_exposure + new_notional
        exposure_pct = total_exposure / equity

        if exposure_pct > Decimal(str(max_exposure_pct)):
            raise ValidationError(
                f"Total exposure {exposure_pct:.2%} would exceed cap of {max_exposure_pct:.0%}"
            )

    def _validate_mmr_projection(
        self, symbol: str, notional: Decimal, equity: Decimal, is_daytime: bool
    ) -> None:
        """Validate that projected liquidation buffer is sufficient."""
        # Get the appropriate MMR based on time of day
        if is_daytime:
            mmr_map = getattr(self.config, "day_mmr_per_symbol", {})
        else:
            mmr_map = getattr(self.config, "night_mmr_per_symbol", {})

        mmr = mmr_map.get(symbol, 0.0)
        if mmr == 0:
            return

        # Calculate projected margin requirement
        margin_required = notional * Decimal(str(mmr))

        # Check against minimum liquidation buffer
        min_buffer = getattr(self.config, "min_liquidation_buffer_pct", 0.0)
        buffer_pct = (equity - margin_required) / equity if equity > 0 else Decimal("0")

        if buffer_pct < Decimal(str(min_buffer)):
            raise ValidationError(
                f"Projected liquidation buffer {buffer_pct:.2%} below minimum {min_buffer:.0%}"
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
            self._save_state()
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
                self._save_state()
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
        avg_close = sum(closes, Decimal("0")) / len(closes)
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
        self._save_state()

    def is_reduce_only_mode(self) -> bool:
        """Check if reduce-only mode is active."""
        return self._reduce_only_mode

    def reset_daily_tracking(self) -> None:
        """Reset daily tracking state (call at start of each trading day)."""
        self._start_of_day_equity = None
        self._daily_pnl_triggered = False
        self._reduce_only_mode = False
        self._reduce_only_reason = ""
        self._save_state()
