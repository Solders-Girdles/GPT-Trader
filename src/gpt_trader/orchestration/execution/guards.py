"""
Runtime guard management for live trading execution.

This module handles all runtime safety checks including daily loss limits,
liquidation buffers, mark price staleness, and volatility circuit breakers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from gpt_trader.features.brokerages.core.interfaces import Balance, IBrokerage
from gpt_trader.features.live_trade.guard_errors import (
    GuardError,
    RiskGuardActionError,
    RiskGuardComputationError,
    RiskGuardDataCorrupt,
    RiskGuardDataUnavailable,
    RiskGuardTelemetryError,
    record_guard_failure,
    record_guard_success,
)
from gpt_trader.features.live_trade.risk import LiveRiskManager
from gpt_trader.monitoring.system import get_logger as _get_plog
from gpt_trader.utilities.logging_patterns import get_logger
from gpt_trader.utilities.quantities import quantity_from

logger = get_logger(__name__, component="execution_guards")


@dataclass
class RuntimeGuardState:
    """Cached snapshot of account risk state used by runtime guards."""

    timestamp: float
    balances: list[Balance]
    equity: Decimal
    positions: list[Any]
    positions_pnl: dict[str, dict[str, Decimal]]
    positions_dict: dict[str, dict[str, Decimal]]
    guard_events: list[dict[str, Any]] = field(default_factory=list)


class GuardManager:
    """Manages runtime safety guards for live trading execution."""

    def __init__(
        self,
        broker: IBrokerage,
        risk_manager: LiveRiskManager,
        equity_calculator: Callable[[list[Balance]], tuple[Decimal, list[Balance], Decimal]],
        open_orders: list[str],
        invalidate_cache_callback: Callable[[], None],
    ) -> None:
        """
        Initialize guard manager.

        Args:
            broker: Brokerage adapter
            risk_manager: Risk manager instance
            equity_calculator: Function to calculate equity from balances
            open_orders: List of open order IDs to manage
            invalidate_cache_callback: Function to invalidate guard cache
        """
        self.broker = broker
        self.risk_manager = risk_manager
        self._calculate_equity = equity_calculator
        self.open_orders = open_orders
        self._invalidate_cache_callback = invalidate_cache_callback

        # Guard state caching
        self._runtime_guard_state: RuntimeGuardState | None = None
        self._runtime_guard_dirty = True
        self._runtime_guard_last_full_ts = 0.0
        self._runtime_guard_full_interval = 60.0

    def invalidate_cache(self) -> None:
        """Force the next runtime guard invocation to refresh cached state."""
        self._runtime_guard_state = None
        self._runtime_guard_dirty = True
        if self._invalidate_cache_callback is not None:
            self._invalidate_cache_callback()

    def should_run_full_guard(self, now: float) -> bool:
        """Check if a full guard run is needed."""
        if self._runtime_guard_dirty:
            return True
        if self._runtime_guard_state is None:
            return True
        return (now - self._runtime_guard_last_full_ts) >= self._runtime_guard_full_interval

    def collect_runtime_guard_state(self) -> RuntimeGuardState:
        """Collect current account state for guard evaluation."""
        balances = self.broker.list_balances()
        equity, _, _ = self._calculate_equity(balances)
        if equity == Decimal("0") and balances:
            equity = sum((b.available for b in balances), Decimal("0"))

        positions = self.broker.list_positions()

        positions_pnl: dict[str, dict[str, Decimal]] = {}
        for pos in positions:
            if hasattr(self.broker, "get_position_pnl"):
                try:
                    pnl_data = self.broker.get_position_pnl(pos.symbol)  # type: ignore[attr-defined]
                    if isinstance(pnl_data, dict):
                        positions_pnl[pos.symbol] = {
                            "realized_pnl": Decimal(str(pnl_data.get("realized_pnl", "0"))),
                            "unrealized_pnl": Decimal(str(pnl_data.get("unrealized_pnl", "0"))),
                        }
                        continue
                except Exception as exc:
                    logger.error(
                        "Failed to get position PnL from broker",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        operation="collect_runtime_guard_state",
                        symbol=pos.symbol,
                    )

            try:
                entry_price = Decimal(str(getattr(pos, "entry_price", "0")))
                mark_price = Decimal(str(getattr(pos, "mark_price", "0")))
                position_quantity = quantity_from(pos) or Decimal("0")
                side = getattr(pos, "side", "").lower()
                side_multiplier = Decimal("1") if side == "long" else Decimal("-1")
                unrealized = (mark_price - entry_price) * position_quantity * side_multiplier
            except Exception as exc:
                logger.error(
                    "Failed to calculate unrealized PnL for position",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="collect_runtime_guard_state",
                    symbol=pos.symbol,
                )
                unrealized = Decimal("0")
            positions_pnl[pos.symbol] = {
                "realized_pnl": Decimal("0"),
                "unrealized_pnl": unrealized,
            }

        positions_dict: dict[str, dict[str, Decimal]] = {}
        for pos in positions:
            try:
                positions_dict[pos.symbol] = {
                    "quantity": quantity_from(pos) or Decimal("0"),
                    "mark": Decimal(str(getattr(pos, "mark_price", "0"))),
                    "entry": Decimal(str(getattr(pos, "entry_price", "0"))),
                }
            except Exception as exc:
                logger.error(
                    "Failed to parse position data for positions_dict",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="collect_runtime_guard_state",
                    symbol=getattr(pos, "symbol", "unknown"),
                )
                continue

        return RuntimeGuardState(
            timestamp=time.time(),
            balances=balances,
            equity=equity,
            positions=positions,
            positions_pnl=positions_pnl,
            positions_dict=positions_dict,
            guard_events=[],
        )

    def run_guard_step(self, guard_name: str, func: Callable[[], None]) -> None:
        """Execute a guard step and apply unified error handling."""
        try:
            func()
        except GuardError as err:
            record_guard_failure(err)
            if err.recoverable:
                return
            raise
        except Exception:
            fallback_err = RiskGuardComputationError(
                guard_name=guard_name,
                message=f"Unexpected failure in guard '{guard_name}'",
                details={},
            )
            record_guard_failure(fallback_err)
            raise fallback_err
        else:
            record_guard_success(guard_name)

    def log_guard_telemetry(self, state: RuntimeGuardState) -> None:
        """Log P&L telemetry for all positions."""
        guard_name = "pnl_telemetry"
        plog = _get_plog()
        failures: list[dict[str, Any]] = []

        for sym, pnl in state.positions_pnl.items():
            rp = pnl.get("realized_pnl")
            up = pnl.get("unrealized_pnl")
            rp_f = float(rp) if rp is not None else None
            up_f = float(up) if up is not None else None
            try:
                plog.log_pnl(symbol=sym, realized_pnl=rp_f, unrealized_pnl=up_f)
            except Exception as exc:
                failures.append({"symbol": sym, "error": repr(exc)})

        if failures:
            raise RiskGuardTelemetryError(
                guard_name=guard_name,
                message="Failed to emit PnL telemetry for one or more symbols",
                details={"failures": failures},
            )

    def guard_daily_loss(self, state: RuntimeGuardState) -> None:
        """Check daily loss limits and cancel orders if breached."""
        guard_name = "daily_loss"

        triggered = self.risk_manager.track_daily_pnl(state.equity, state.positions_pnl)
        if triggered:
            try:
                self.cancel_all_orders()
            except Exception as exc:
                raise RiskGuardActionError(
                    guard_name=guard_name,
                    message="Failed to cancel orders after daily loss breach",
                    details={"equity": str(state.equity)},
                ) from exc
            self.invalidate_cache()

    def guard_liquidation_buffers(self, state: RuntimeGuardState, incremental: bool) -> None:
        """Check liquidation price buffers for all positions."""
        guard_name = "liquidation_buffer"

        for pos in state.positions:
            try:
                position_quantity = quantity_from(pos) or Decimal("0")
                mark = Decimal(str(getattr(pos, "mark_price", "0")))
            except Exception as exc:
                raise RiskGuardDataCorrupt(
                    guard_name=guard_name,
                    message="Position payload missing numeric fields",
                    details={"symbol": getattr(pos, "symbol", "unknown")},
                ) from exc

            pos_data: dict[str, Any] = {
                "quantity": position_quantity,
                "mark": mark,
            }

            if not incremental and hasattr(self.broker, "get_position_risk"):
                try:
                    risk_info = self.broker.get_position_risk(pos.symbol)  # type: ignore[attr-defined]
                except Exception as exc:
                    raise RiskGuardDataUnavailable(
                        guard_name=guard_name,
                        message="Failed to fetch position risk from broker",
                        details={"symbol": pos.symbol},
                    ) from exc
                if isinstance(risk_info, dict) and "liquidation_price" in risk_info:
                    pos_data["liquidation_price"] = risk_info["liquidation_price"]

            self.risk_manager.check_liquidation_buffer(pos.symbol, pos_data, state.equity)

    def guard_mark_staleness(self, state: RuntimeGuardState) -> None:
        """Check if mark prices are stale."""
        guard_name = "mark_staleness"
        if not hasattr(self.broker, "_mark_cache"):
            return

        failures: list[dict[str, Any]] = []
        for symbol in list(self.risk_manager.last_mark_update.keys()):
            try:
                mark = self.broker._mark_cache.get_mark(symbol)  # type: ignore[attr-defined]
            except Exception as exc:
                failures.append({"symbol": symbol, "error": repr(exc)})
                continue
            if mark is None:
                self.risk_manager.check_mark_staleness(symbol)

        if failures:
            raise RiskGuardDataUnavailable(
                guard_name=guard_name,
                message="Failed to refresh mark data for one or more symbols",
                details={"failures": failures},
            )

    def guard_risk_metrics(self, state: RuntimeGuardState) -> None:
        """Append risk metrics for monitoring."""
        guard_name = "risk_metrics"
        try:
            self.risk_manager.append_risk_metrics(state.equity, state.positions_dict)
        except GuardError as err:
            raise err
        except Exception as exc:
            raise RiskGuardTelemetryError(
                guard_name=guard_name,
                message="Failed to append risk metrics",
                details={"equity": str(state.equity)},
            ) from exc

    def guard_correlation(self, state: RuntimeGuardState) -> None:
        """Check correlation risk across positions."""
        guard_name = "correlation_risk"
        try:
            self.risk_manager.check_correlation_risk(state.positions_dict)
        except GuardError as err:
            raise err
        except Exception as exc:
            raise RiskGuardComputationError(
                guard_name=guard_name,
                message="Correlation risk check failed",
                details={"positions": list(state.positions_dict.keys())},
            ) from exc

    def guard_volatility(self, state: RuntimeGuardState) -> None:
        """Check volatility circuit breakers."""
        guard_name = "volatility_circuit_breaker"
        symbols: list[str] = list(self.risk_manager.last_mark_update.keys())
        symbols.extend([str(p) for p in state.positions_dict.keys() if p not in symbols])
        window = getattr(self.risk_manager.config, "volatility_window_periods", 20)
        if not symbols or not window or window <= 5:
            return

        failures: list[dict[str, Any]] = []
        for sym in symbols:
            if not hasattr(self.broker, "get_candles"):
                continue
            try:
                candles = self.broker.get_candles(sym, granularity="1m", limit=int(window))
            except Exception as exc:
                failures.append({"symbol": sym, "error": repr(exc)})
                continue
            closes = [c.close for c in candles if hasattr(c, "close")]
            if len(closes) >= window:
                outcome = self.risk_manager.check_volatility_circuit_breaker(sym, closes[-window:])
                if outcome.triggered:
                    state.guard_events.append(outcome.to_payload())

        if failures:
            raise RiskGuardDataUnavailable(
                guard_name=guard_name,
                message="Failed to fetch candles for volatility guard",
                details={"failures": failures},
            )

    def run_guards_for_state(self, state: RuntimeGuardState, incremental: bool) -> None:
        """Run all guards for the given state."""
        self.run_guard_step("pnl_telemetry", lambda: self.log_guard_telemetry(state))
        self.run_guard_step("daily_loss", lambda: self.guard_daily_loss(state))
        self.run_guard_step(
            "liquidation_buffer",
            lambda: self.guard_liquidation_buffers(state, incremental),
        )
        self.run_guard_step("mark_staleness", lambda: self.guard_mark_staleness(state))
        self.run_guard_step("risk_metrics", lambda: self.guard_risk_metrics(state))
        self.run_guard_step("correlation_risk", lambda: self.guard_correlation(state))
        self.run_guard_step("volatility_circuit_breaker", lambda: self.guard_volatility(state))

    def run_runtime_guards(self, force_full: bool = False) -> RuntimeGuardState:
        """
        Execute runtime guards and return the guard state.

        Args:
            force_full: Force a full guard run even if cached

        Returns:
            RuntimeGuardState snapshot after guard execution
        """
        now = time.time()
        incremental = not force_full and not self.should_run_full_guard(now)

        if not incremental or self._runtime_guard_state is None:
            state = self.collect_runtime_guard_state()
            self._runtime_guard_state = state
            self._runtime_guard_last_full_ts = now
            self._runtime_guard_dirty = False
        else:
            state = self._runtime_guard_state

        self.run_guards_for_state(state, incremental)
        return state

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders (used on risk trips).

        Returns:
            Number of orders cancelled
        """
        cancelled = 0

        for order_id in self.open_orders[:]:  # Copy list to avoid modification during iteration
            try:
                if self.broker.cancel_order(order_id):
                    cancelled += 1
                    self.open_orders.remove(order_id)
                    logger.info(
                        "Cancelled order",
                        order_id=order_id,
                        operation="order_cancel",
                        stage="single",
                    )
            except Exception as e:
                logger.error(
                    "Failed to cancel order %s: %s",
                    order_id,
                    e,
                    operation="order_cancel",
                    stage="single",
                    order_id=order_id,
                )

        if cancelled > 0:
            logger.info(
                "Cancelled open orders due to risk trip",
                cancelled=cancelled,
                operation="order_cancel",
                stage="bulk",
            )
            self.invalidate_cache()

        return cancelled

    def safe_run_runtime_guards(self, force_full: bool = False) -> None:
        """
        Run runtime risk guards and take action if needed.
        Handles exceptions and fallback logic.
        """
        try:
            self.run_runtime_guards(force_full=force_full)

        except GuardError as err:
            level = logging.WARNING if err.recoverable else logging.ERROR
            logger.log(
                level,
                "Runtime guard failure: %s",
                err,
                exc_info=not err.recoverable,
                operation="runtime_guards",
                stage="guard_failure",
                recoverable=err.recoverable,
            )
            if not err.recoverable:
                try:
                    # Fallback to legacy behavior
                    self.risk_manager.set_reduce_only_mode(True, reason="guard_failure")
                except Exception:
                    logger.warning(
                        "Failed to set reduce-only mode after guard failure",
                        exc_info=True,
                        operation="runtime_guards",
                        stage="reduce_only",
                    )
                self.invalidate_cache()

        except Exception as exc:
            logger.error(
                "Runtime guards error: %s",
                exc,
                operation="runtime_guards",
                stage="exception",
            )
