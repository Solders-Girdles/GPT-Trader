"""
Runtime guards and continuous monitoring for risk management.

Async and periodic checks performed during trading operations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
from datetime import datetime
from decimal import Decimal
from typing import Any

from bot_v2.features.live_trade.guard_errors import (
    RiskGuardComputationError,
    RiskGuardDataCorrupt,
    RiskGuardTelemetryError,
)
from bot_v2.features.live_trade.risk_runtime.circuit_breakers import (
    CircuitBreakerOutcome,
    CircuitBreakerState,
)
from bot_v2.features.live_trade.risk_runtime.circuit_breakers import (
    check_volatility_circuit_breaker as runtime_check_volatility_circuit_breaker,
)
from bot_v2.features.live_trade.risk_runtime.guards import (
    check_correlation_risk as runtime_check_correlation_risk,
)
from bot_v2.features.live_trade.risk_runtime.guards import (
    check_mark_staleness as runtime_check_mark_staleness,
)
from bot_v2.features.live_trade.risk_runtime.metrics import (
    append_risk_metrics as runtime_append_risk_metrics,
)
from bot_v2.orchestration.configuration import RiskConfig
from bot_v2.orchestration.state_manager import ReduceOnlyModeSource
from bot_v2.persistence.event_store import EventStore
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

logger = get_logger(__name__, component="live_trade_risk")


class RuntimeMonitor:
    """Monitors runtime conditions and triggers guards."""

    def __init__(
        self,
        config: RiskConfig,
        event_store: EventStore,
        set_reduce_only_mode: Callable[[bool, str], None] | None = None,
        now_provider: Callable[[], datetime] | None = None,
        last_mark_update: MutableMapping[str, datetime | None] | None = None,
    ):
        """
        Initialize runtime monitor.

        Args:
            config: Risk configuration
            event_store: Event store for risk events
            set_reduce_only_mode: Callable to set reduce-only mode
            now_provider: Time provider for testability
            last_mark_update: Shared dict tracking last mark updates
        """
        self.config = config
        self.event_store = event_store
        self._set_reduce_only_mode = set_reduce_only_mode or (lambda enabled, reason: None)
        self._centralized_state_manager = None  # Will be injected if available
        self._now_provider = now_provider or (lambda: datetime.utcnow())
        if last_mark_update is not None:
            self.last_mark_update: dict[str, datetime] = {
                symbol: timestamp for symbol, timestamp in last_mark_update.items() if timestamp
            }
        else:
            self.last_mark_update = {}

        # Circuit breaker state
        self._cb_last_trigger: dict[str, datetime] = {}
        self.circuit_breaker_state = CircuitBreakerState()

        # Position tracking for reduce-only flags
        self.positions: dict[str, dict[str, Any]] = {}

    def track_daily_pnl(
        self,
        current_equity: Decimal,
        positions_pnl: dict[str, dict[str, Decimal]],
        daily_pnl: Decimal,
        start_of_day_equity: Decimal,
    ) -> tuple[bool, Decimal]:
        """
        Track daily PnL and trigger reduce-only if breaching limit.

        Args:
            current_equity: Current account equity
            positions_pnl: PnL data from Phase 4 states
            daily_pnl: Current daily PnL tracker
            start_of_day_equity: Starting equity for the day

        Returns:
            Tuple of (reduce_only_triggered, updated_daily_pnl)
        """
        # Calculate total PnL
        total_realized = Decimal("0")
        total_unrealized = Decimal("0")

        for symbol, pnl_data in positions_pnl.items():
            total_realized += pnl_data.get("realized_pnl", Decimal("0"))
            total_unrealized += pnl_data.get("unrealized_pnl", Decimal("0"))

        daily_pnl = total_realized + total_unrealized

        # Check daily loss limit
        daily_loss_abs = -daily_pnl

        if daily_loss_abs > self.config.daily_loss_limit:
            # Trip reduce-only mode
            # Try to use the centralized state manager first
            if self._centralized_state_manager is not None:
                self._centralized_state_manager.set_reduce_only_mode(
                    enabled=True,
                    reason="daily_loss_limit",
                    source=ReduceOnlyModeSource.DAILY_LOSS_LIMIT,
                    metadata={"loss_amount": str(daily_loss_abs)},
                )
            else:
                # Fallback to legacy behavior
                self._set_reduce_only_mode(True, "daily_loss_limit")

            # Log risk event
            self._log_risk_event(
                "daily_loss_breach",
                {
                    "daily_pnl": str(daily_pnl),
                    "daily_loss_abs": str(daily_loss_abs),
                    "limit": str(self.config.daily_loss_limit),
                    "action": "reduce_only_mode_enabled",
                },
                guard="daily_loss",
            )

            logger.warning(
                f"Daily loss limit breached: ${daily_loss_abs} > ${self.config.daily_loss_limit} - Enabling reduce-only mode"
            )

            return True, daily_pnl

        return False, daily_pnl

    def check_liquidation_buffer(
        self, symbol: str, position_data: dict[str, Any], equity: Decimal
    ) -> bool:
        """
        Monitor liquidation buffer for position.

        Returns:
            True if reduce-only was set for this symbol
        """
        guard_name = "liquidation_buffer"

        try:
            qty = Decimal(str(position_data.get("quantity", position_data.get("qty", 0))))
            mark = Decimal(str(position_data.get("mark", 0)))
        except Exception as exc:
            raise RiskGuardDataCorrupt(
                guard=guard_name,
                message="Invalid position data for liquidation buffer",
                details={"symbol": symbol, "position_data": position_data},
                original=exc,
            ) from exc

        # Optional real liquidation price from exchange
        liq_raw = position_data.get("liquidation_price") or position_data.get("liq_price")
        liquidation_price: Decimal | None = None
        try:
            if liq_raw is not None:
                liquidation_price = Decimal(str(liq_raw))
        except Exception as exc:
            raise RiskGuardDataCorrupt(
                guard=guard_name,
                message="Invalid liquidation price",
                details={"symbol": symbol, "liquidation_price": liq_raw},
                original=exc,
            ) from exc

        if qty == 0 or mark == 0:
            return False

        notional = abs(qty * mark)

        # If we have a true liquidation price, compute distance-to-liquidation as buffer
        buffer_pct = Decimal("0")
        if liquidation_price is not None and mark > 0:
            try:
                buffer_pct = abs(mark - liquidation_price) / mark
            except Exception as exc:
                raise RiskGuardComputationError(
                    guard=guard_name,
                    message="Failed to compute buffer using liquidation price",
                    details={"symbol": symbol},
                    original=exc,
                ) from exc
        else:
            # Fallback: estimate via leverage
            max_leverage = self.config.leverage_max_per_symbol.get(symbol, self.config.max_leverage)
            try:
                max_leverage_decimal = (
                    max_leverage
                    if isinstance(max_leverage, Decimal)
                    else Decimal(str(max_leverage))
                )
                margin_used = (
                    notional / max_leverage_decimal if max_leverage_decimal > 0 else notional
                )
                buffer_pct = (equity - margin_used) / equity if equity > 0 else Decimal("0")
            except Exception as exc:
                raise RiskGuardComputationError(
                    guard=guard_name,
                    message="Failed to compute buffer using leverage fallback",
                    details={"symbol": symbol},
                    original=exc,
                ) from exc

        if buffer_pct < self.config.min_liquidation_buffer_pct:
            # Set reduce-only for this symbol
            if symbol not in self.positions:
                self.positions[symbol] = {}
            self.positions[symbol]["reduce_only"] = True

            # Log risk event
            self._log_risk_event(
                "liquidation_buffer_breach",
                {
                    "symbol": symbol,
                    "buffer_pct": str(buffer_pct),
                    "limit": str(self.config.min_liquidation_buffer_pct),
                    "action": f"reduce_only_enabled_for_{symbol}",
                },
                guard=guard_name,
            )

            logger.warning(
                f"Liquidation buffer breach for {symbol}: {buffer_pct:.2%} < "
                f"{self.config.min_liquidation_buffer_pct:.2%} - Setting reduce-only"
            )

            return True

        return False

    def check_mark_staleness(self, symbol: str, mark_timestamp: datetime | None = None) -> bool:
        """Check whether mark data for symbol is stale."""
        if mark_timestamp is not None:
            self.last_mark_update[symbol] = mark_timestamp
        return bool(
            runtime_check_mark_staleness(
                symbol=symbol,
                last_mark_update=self.last_mark_update,
                now=self._now_provider,
                max_staleness_seconds=self.config.max_mark_staleness_seconds,
                log_event=self._log_risk_event,
                logger=logger,
            )
        )

    def append_risk_metrics(
        self,
        equity: Decimal,
        positions: dict[str, Any],
        daily_pnl: Decimal,
        start_of_day_equity: Decimal,
        is_reduce_only_mode: bool,
    ) -> None:
        """Append periodic risk metrics snapshot to the event store."""
        runtime_append_risk_metrics(
            event_store=self.event_store,
            now=self._now_provider,
            equity=equity,
            positions=positions,
            daily_pnl=daily_pnl,
            start_of_day_equity=start_of_day_equity,
            reduce_only=is_reduce_only_mode,
            kill_switch_enabled=self.config.kill_switch_enabled,
            logger=logger,
        )

    def check_correlation_risk(self, positions: dict[str, Any]) -> bool:
        """Check portfolio correlation and concentration risk."""
        return bool(
            runtime_check_correlation_risk(
                positions,
                log_event=self._log_risk_event,
                logger=logger,
            )
        )

    def check_volatility_circuit_breaker(
        self, symbol: str, recent_marks: list[Decimal]
    ) -> CircuitBreakerOutcome:
        """Check rolling volatility and trigger progressive circuit breakers."""
        outcome = runtime_check_volatility_circuit_breaker(
            symbol=symbol,
            recent_marks=recent_marks,
            config=self.config,
            state=self.circuit_breaker_state,
            now=self._now_provider,
            last_trigger=self._cb_last_trigger,
            set_reduce_only=lambda enabled, reason: self._set_reduce_only_mode(enabled, reason),
            log_event=self._log_risk_event,
            centralized_state_manager=self._centralized_state_manager,
            logger=logger,
        )

        if outcome.triggered:
            try:
                self.circuit_breaker_state.record(
                    "volatility_circuit_breaker",
                    symbol,
                    outcome.action,
                    self._now_provider(),
                )
            except Exception:
                logger.debug("Failed to record circuit breaker snapshot", exc_info=True)

        return outcome

    def _log_risk_event(
        self, event_type: str, details: Mapping[str, Any], guard: str | None = None
    ) -> None:
        """Log risk event to EventStore, surfacing telemetry failures."""
        guard_name = guard or event_type
        try:
            emit_metric(
                self.event_store,
                "risk_engine",
                {
                    "event_type": event_type,
                    "timestamp": datetime.utcnow().isoformat(),
                    **dict(details),
                },
                logger=logger,
                raise_on_error=True,
            )
        except Exception as exc:
            raise RiskGuardTelemetryError(
                guard=guard_name,
                message=f"Failed to persist risk event '{event_type}'",
                details=dict(details),
                original=exc,
            ) from exc
