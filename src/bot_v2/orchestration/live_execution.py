"""
Live execution engine with risk management integration.

Phase 5: Risk engine integration for perpetuals.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from bot_v2.monitoring.system import LogLevel, get_logger
from bot_v2.utilities.quantities import quantity_from

from ..features.brokerages.coinbase.specs import (
    validate_order as spec_validate_order,
)
from ..features.brokerages.core.interfaces import (
    Balance,
    IBrokerage,
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from ..features.live_trade.guard_errors import (
    RiskGuardActionError,
    RiskGuardComputationError,
    RiskGuardDataCorrupt,
    RiskGuardDataUnavailable,
    RiskGuardError,
    RiskGuardTelemetryError,
    record_guard_failure,
    record_guard_success,
)
from ..features.live_trade.risk import LiveRiskManager, ValidationError
from ..persistence.event_store import EventStore
from ..utilities.quantization import quantize_price_side_aware

logger = logging.getLogger(__name__)


@dataclass
class LiveOrder:
    """Live order details."""

    symbol: str
    side: str  # "buy" or "sell"
    quantity: Decimal
    price: Decimal | None = None  # None for market orders
    order_type: str = "market"
    reduce_only: bool = False
    leverage: int | None = None

    def __post_init__(self) -> None:
        self.quantity = Decimal(str(self.quantity))


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


class LiveExecutionEngine:
    """Live execution with integrated risk controls for perpetuals.

    Enforces risk checks before order placement and monitors runtime guards.
    """

    def __init__(
        self,
        broker: IBrokerage,
        risk_manager: LiveRiskManager | None = None,
        event_store: EventStore | None = None,
        bot_id: str = "live_execution",
        slippage_multipliers: dict[str, float] | None = None,
        enable_preview: bool | None = None,
    ) -> None:
        """
        Initialize live execution engine.

        Args:
            broker: Brokerage adapter (must support perpetuals)
            risk_manager: Risk manager instance (creates default if None)
            event_store: Event store for metrics
            bot_id: Bot identifier for logging
        """
        self.broker = broker
        self.risk_manager = risk_manager or LiveRiskManager()
        self.event_store = event_store or EventStore()
        self.bot_id = bot_id
        self.slippage_multipliers = slippage_multipliers or {}
        self.collateral_assets: set[str] = self._resolve_collateral_assets()
        preview_env = os.getenv("ORDER_PREVIEW_ENABLED")
        if enable_preview is not None:
            self.enable_order_preview = enable_preview
        elif preview_env is not None:
            self.enable_order_preview = preview_env.lower() in ("1", "true", "yes")
        else:
            self.enable_order_preview = False

        # Track open orders for cancellation on risk trips
        self.open_orders: list[str] = []
        # Track last seen collateral availability for balance change logs
        self._last_collateral_available: Decimal | None = None

        # Cache runtime guard state to avoid redundant API calls on every invocation
        self._runtime_guard_state: RuntimeGuardState | None = None
        self._runtime_guard_dirty: bool = False
        self._runtime_guard_last_run_ts: float = 0.0
        self._runtime_guard_last_full_ts: float = 0.0
        try:
            default_interval = 45
            env_value = int(os.getenv("RUNTIME_GUARD_FULL_INTERVAL_SEC", default_interval))
            self._runtime_guard_full_interval = max(15, env_value)
        except Exception:
            self._runtime_guard_full_interval = 45

        logger.info(f"LiveExecutionEngine initialized for {bot_id}")

    def _resolve_collateral_assets(self) -> set[str]:
        env_value = os.getenv("PERPS_COLLATERAL_ASSETS") or ""
        default_assets = {"USD", "USDC"}
        parsed = {token.strip().upper() for token in env_value.split(",") if token.strip()}
        return parsed or set(default_assets)

    def _calculate_equity_from_balances(
        self,
        balances: list[Balance],
    ) -> tuple[Decimal, list[Balance], Decimal]:
        total_available = Decimal("0")
        total_balance = Decimal("0")
        collateral_balances: list[Balance] = []

        for bal in balances:
            asset = (bal.asset or "").upper()
            if asset in self.collateral_assets:
                collateral_balances.append(bal)
                total_available += bal.available
                total_balance += bal.total

        if collateral_balances:
            return total_available, collateral_balances, total_balance

        usd_balance = next((bal for bal in balances if (bal.asset or "").upper() == "USD"), None)
        if usd_balance:
            return usd_balance.available, [usd_balance], usd_balance.total

        return Decimal("0"), [], Decimal("0")

    def _invalidate_runtime_guard_cache(self) -> None:
        """Force the next runtime guard invocation to refresh cached state."""

        self._runtime_guard_state = None
        self._runtime_guard_dirty = True

    def _should_run_full_runtime_guard(self, now: float) -> bool:
        if self._runtime_guard_dirty:
            return True
        if self._runtime_guard_state is None:
            return True
        return (now - self._runtime_guard_last_full_ts) >= self._runtime_guard_full_interval

    def _collect_runtime_guard_state(self) -> RuntimeGuardState:
        balances = self.broker.list_balances()
        equity, _, _ = self._calculate_equity_from_balances(balances)
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
                except Exception:
                    pass

            try:
                entry_price = Decimal(str(getattr(pos, "entry_price", "0")))
                mark_price = Decimal(str(getattr(pos, "mark_price", "0")))
                position_quantity = quantity_from(pos) or Decimal("0")
                side = getattr(pos, "side", "").lower()
                side_multiplier = Decimal("1") if side == "long" else Decimal("-1")
                unrealized = (mark_price - entry_price) * position_quantity * side_multiplier
            except Exception:
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
            except Exception:
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

    def _run_guard_step(self, guard_name: str, func: Callable[[], None]) -> None:
        """Execute a guard step and apply unified error handling."""

        try:
            func()
        except RiskGuardError as err:
            record_guard_failure(err)
            if err.recoverable:
                return
            raise
        except Exception as exc:
            err = RiskGuardComputationError(
                guard=guard_name,
                message=f"Unexpected failure in guard '{guard_name}'",
                details={},
                original=exc,
            )
            record_guard_failure(err)
            raise err
        else:
            record_guard_success(guard_name)

    def _log_guard_telemetry(self, state: RuntimeGuardState) -> None:
        guard_name = "pnl_telemetry"
        plog = get_logger()
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
                guard=guard_name,
                message="Failed to emit PnL telemetry for one or more symbols",
                details={"failures": failures},
            )

    def _guard_daily_loss(self, state: RuntimeGuardState) -> None:
        guard_name = "daily_loss"

        triggered = self.risk_manager.track_daily_pnl(state.equity, state.positions_pnl)
        if triggered:
            try:
                self.cancel_all_orders()
            except Exception as exc:
                raise RiskGuardActionError(
                    guard=guard_name,
                    message="Failed to cancel orders after daily loss breach",
                    details={"equity": str(state.equity)},
                    original=exc,
                ) from exc
            self._invalidate_runtime_guard_cache()

    def _guard_liquidation_buffers(self, state: RuntimeGuardState, incremental: bool) -> None:
        guard_name = "liquidation_buffer"

        for pos in state.positions:
            try:
                position_quantity = quantity_from(pos) or Decimal("0")
                mark = Decimal(str(getattr(pos, "mark_price", "0")))
            except Exception as exc:
                raise RiskGuardDataCorrupt(
                    guard=guard_name,
                    message="Position payload missing numeric fields",
                    details={"symbol": getattr(pos, "symbol", "unknown")},
                    original=exc,
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
                        guard=guard_name,
                        message="Failed to fetch position risk from broker",
                        details={"symbol": pos.symbol},
                        original=exc,
                    ) from exc
                if isinstance(risk_info, dict) and "liquidation_price" in risk_info:
                    pos_data["liquidation_price"] = risk_info["liquidation_price"]

            self.risk_manager.check_liquidation_buffer(pos.symbol, pos_data, state.equity)

    def _guard_mark_staleness(self, state: RuntimeGuardState) -> None:
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
                guard=guard_name,
                message="Failed to refresh mark data for one or more symbols",
                details={"failures": failures},
            )

    def _guard_risk_metrics(self, state: RuntimeGuardState) -> None:
        guard_name = "risk_metrics"
        try:
            self.risk_manager.append_risk_metrics(state.equity, state.positions_dict)
        except RiskGuardError as err:
            raise err
        except Exception as exc:
            raise RiskGuardTelemetryError(
                guard=guard_name,
                message="Failed to append risk metrics",
                details={"equity": str(state.equity)},
                original=exc,
            ) from exc

    def _guard_correlation(self, state: RuntimeGuardState) -> None:
        guard_name = "correlation_risk"
        try:
            self.risk_manager.check_correlation_risk(state.positions_dict)
        except RiskGuardError as err:
            raise err
        except Exception as exc:
            raise RiskGuardComputationError(
                guard=guard_name,
                message="Correlation risk check failed",
                details={"positions": list(state.positions_dict.keys())},
                original=exc,
            ) from exc

    def _guard_volatility(self, state: RuntimeGuardState) -> None:
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
                guard=guard_name,
                message="Failed to fetch candles for volatility guard",
                details={"failures": failures},
            )

    def _run_runtime_guards_for_state(self, state: RuntimeGuardState, incremental: bool) -> None:
        self._run_guard_step("pnl_telemetry", lambda: self._log_guard_telemetry(state))
        self._run_guard_step("daily_loss", lambda: self._guard_daily_loss(state))
        self._run_guard_step(
            "liquidation_buffer",
            lambda: self._guard_liquidation_buffers(state, incremental),
        )
        self._run_guard_step("mark_staleness", lambda: self._guard_mark_staleness(state))
        self._run_guard_step("risk_metrics", lambda: self._guard_risk_metrics(state))
        self._run_guard_step("correlation_risk", lambda: self._guard_correlation(state))

        if not incremental:
            self._run_guard_step(
                "volatility_circuit_breaker", lambda: self._guard_volatility(state)
            )

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal | None = None,
        price: Decimal | None = None,
        stop_price: Decimal | None = None,
        tif: Any | None = None,
        reduce_only: bool = False,
        leverage: int | None = None,
        product: Product | None = None,
        client_order_id: str | None = None,
    ) -> str | None:
        """
        Place order with pre-trade risk validation.

        Args:
            symbol: Trading symbol
            side: OrderSide enum
            order_type: OrderType enum
            quantity: Order quantity
            price: Limit price (None for market)
            reduce_only: Force reduce-only order
            leverage: Target leverage for perpetuals
            product: Product metadata (fetched if None)

        Returns:
            Order ID if successful, None if rejected

        Raises:
            ValidationError: If risk checks fail
        """
        if quantity is None:
            raise TypeError("place_order requires 'quantity'")

        order_quantity = Decimal(str(quantity))
        effective_price: Decimal | None = None
        price_decimal: Decimal | None = Decimal(str(price)) if price is not None else None
        try:
            product = self._require_product(symbol, product)

            (
                balances,
                equity,
                collateral_balances,
                collateral_total,
                current_positions,
            ) = self._collect_account_state()

            self._log_collateral_update(collateral_balances, equity, collateral_total, balances)

            effective_price = self._resolve_effective_price(symbol, side, order_type, price_decimal)

            order_quantity, price_decimal = self._validate_exchange_rules(
                symbol,
                side,
                order_type,
                order_quantity,
                price_decimal,
                effective_price,
                product,
            )

            self._ensure_mark_is_fresh(symbol)
            self._enforce_slippage_guard(symbol, side, order_quantity, effective_price)

            self._run_pre_trade_validation(
                symbol=symbol,
                side=side,
                order_quantity=order_quantity,
                effective_price=effective_price,
                product=product,
                equity=equity,
                current_positions=current_positions,
            )

            self._maybe_preview_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                order_quantity=order_quantity,
                effective_price=effective_price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=reduce_only,
                leverage=leverage,
            )

            is_reduce_only = self._finalize_reduce_only_flag(reduce_only, symbol)

            return self._submit_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                order_quantity=order_quantity,
                price=price_decimal,
                effective_price=effective_price,
                stop_price=stop_price,
                tif=tif,
                reduce_only=is_reduce_only,
                leverage=leverage,
                client_order_id=client_order_id,
            )
        except ValidationError as e:
            logger.warning(f"Risk validation failed: {e}")
            rejection_price = price_decimal if price_decimal is not None else effective_price
            self._record_rejection(symbol, side.value, order_quantity, rejection_price, str(e))
            raise

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            try:
                self.event_store.append_error(
                    bot_id=self.bot_id,
                    message="order_exception",
                    context={
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": str(order_quantity),
                        "error": str(e),
                    },
                )
            except Exception:
                pass
            return None
        finally:
            self._invalidate_runtime_guard_cache()

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
                    logger.info(f"Cancelled order: {order_id}")
            except Exception as e:
                logger.error(f"Failed to cancel order {order_id}: {e}")

        if cancelled > 0:
            logger.info(f"Cancelled {cancelled} open orders due to risk trip")
            self._invalidate_runtime_guard_cache()

        return cancelled

    def run_runtime_guards(self) -> None:
        """
        Run runtime risk guards and take action if needed.

        Should be called periodically (e.g., every minute).
        """
        try:
            now = time.time()
            if self._should_run_full_runtime_guard(now):
                state = self._collect_runtime_guard_state()
                self._runtime_guard_state = state
                self._runtime_guard_last_full_ts = state.timestamp
                self._runtime_guard_dirty = False
                self._run_runtime_guards_for_state(state, incremental=False)
            else:
                state = self._runtime_guard_state
                if state is None:
                    state = self._collect_runtime_guard_state()
                    self._runtime_guard_state = state
                    self._runtime_guard_last_full_ts = state.timestamp
                    self._runtime_guard_dirty = False
                    self._run_runtime_guards_for_state(state, incremental=False)
                else:
                    self._run_runtime_guards_for_state(state, incremental=True)
            self._runtime_guard_last_run_ts = now
        except RiskGuardError as err:
            level = logging.WARNING if err.recoverable else logging.ERROR
            logger.log(
                level,
                "Runtime guard failure: %s",
                err,
                exc_info=not err.recoverable,
                extra={
                    "guard_failure": err.failure.as_log_args(),
                },
            )
            if not err.recoverable:
                try:
                    self.risk_manager.set_reduce_only_mode(True, reason="guard_failure")
                except Exception:
                    logger.warning(
                        "Failed to set reduce-only mode after guard failure", exc_info=True
                    )
                self._invalidate_runtime_guard_cache()
        except Exception as e:
            logger.error(f"Runtime guards error: {e}")

    def reset_daily_tracking(self) -> None:
        """Reset daily PnL tracking (call at start of trading day)."""
        try:
            # Prefer adapter-neutral equity from balances; fall back to get_account if available
            equity: Decimal | None = None
            try:
                balances = self.broker.list_balances()
                equity_candidate, _, _ = self._calculate_equity_from_balances(balances)
                if equity_candidate > Decimal("0"):
                    equity = equity_candidate
                elif balances:
                    fallback_total = sum((b.available for b in balances), Decimal("0"))
                    if fallback_total > Decimal("0"):
                        equity = fallback_total
            except Exception:
                pass

            if equity is None and hasattr(self.broker, "get_account_snapshot"):
                snapshot = self.broker.get_account_snapshot()  # type: ignore[attr-defined]
                if snapshot is not None:
                    equity = snapshot.equity

            equity = equity if equity is not None else Decimal("0")
            self.risk_manager.reset_daily_tracking(equity)
            logger.info("Daily tracking reset")
        except Exception as e:
            logger.error(f"Failed to reset daily tracking: {e}")

    # ===== Internal helpers =====
    def _record_preview(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Decimal | None,
        preview: dict[str, Any] | None,
    ) -> None:
        if preview is None:
            return
        try:
            self.event_store.append_metric(
                bot_id=self.bot_id,
                metrics={
                    "event_type": "order_preview",
                    "symbol": symbol,
                    "side": side.value,
                    "order_type": order_type.value,
                    "quantity": str(quantity),
                    "price": str(price) if price is not None else "market",
                    "preview": preview,
                },
            )
        except Exception:
            pass
        try:
            get_logger().log_event(
                level=LogLevel.INFO,
                event_type="order_preview",
                message="Order preview generated",
                component="LiveExecutionEngine",
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
            )
        except Exception:
            pass

    def _record_rejection(
        self, symbol: str, side: str, quantity: Decimal, price: Decimal | None, reason: str
    ) -> None:
        logger.warning(
            f"Order rejected: {symbol} {side} {quantity} @ {price or 'market'} reason={reason}"
        )
        # Persist an order_rejected metric for downstream analysis/tests
        try:
            self.event_store._write(
                {
                    "type": "order_rejected",
                    "bot_id": self.bot_id,
                    "symbol": symbol,
                    "side": side,
                    "quantity": str(quantity),
                    "price": str(price) if price is not None else "market",
                    "reason": reason,
                }
            )
        except Exception:
            pass
        try:
            from bot_v2.monitoring.system import get_logger as _get_plog

            _get_plog().log_order_status_change(
                order_id="",
                client_order_id="",
                from_status=None,
                to_status="REJECTED",
                reason=reason,
            )
        except Exception:
            pass

    def _require_product(self, symbol: str, product: Product | None) -> Product:
        if product is not None:
            return product
        fetched = self.broker.get_product(symbol)
        if fetched is None:
            raise ValidationError(f"Product not found: {symbol}")
        return fetched

    def _collect_account_state(
        self,
    ) -> tuple[
        list[Balance],
        Decimal,
        list[Balance],
        Decimal,
        dict[str, dict[str, Any]],
    ]:
        balances = self.broker.list_balances()
        equity, collateral_balances, collateral_total = self._calculate_equity_from_balances(
            balances
        )
        positions = self.broker.list_positions()
        current_positions = self._build_positions_dict(positions)
        return balances, equity, collateral_balances, collateral_total, current_positions

    def _build_positions_dict(self, positions: list[Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for pos in positions:
            symbol_key = getattr(pos, "symbol", None)
            if not symbol_key:
                continue
            pos_quantity = quantity_from(pos) or Decimal("0")
            result[symbol_key] = {
                "quantity": pos_quantity,
                "side": getattr(pos, "side", ""),
                "mark": getattr(pos, "mark_price", None),
            }
        return result

    def _log_collateral_update(
        self,
        collateral_balances: list[Balance],
        equity: Decimal,
        collateral_total: Decimal,
        balances: list[Balance],
    ) -> None:
        try:
            if collateral_balances:
                assets_label = "/".join(
                    sorted({(bal.asset or "").upper() for bal in collateral_balances})
                )
                previous = self._last_collateral_available
                self._last_collateral_available = equity
                change = (equity - previous) if previous is not None else None
                get_logger().log_balance_update(
                    currency=assets_label,
                    available=float(equity),
                    total=float(collateral_total),
                    change=float(change) if change is not None else None,
                    reason="runtime_guard_check",
                )
            elif balances:
                total_available = sum((bal.available for bal in balances), Decimal("0"))
                total_balance = sum((bal.total for bal in balances), Decimal("0"))
                get_logger().log_balance_update(
                    currency="TOTAL",
                    available=float(total_available),
                    total=float(total_balance),
                    reason="runtime_guard_check",
                )
        except Exception:
            pass

    def _resolve_effective_price(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        price: Decimal | None,
    ) -> Decimal:
        if price is not None and order_type != OrderType.MARKET:
            return Decimal(str(price))

        quote = self.broker.get_quote(symbol)
        if not quote:
            raise ValidationError(f"No quote available for {symbol}")

        base_price = quote.ask if side == OrderSide.BUY else quote.bid
        if base_price is None:
            raise ValidationError(f"No valid price available for {symbol}")

        execution_price = Decimal(str(base_price))
        try:
            mult = Decimal(str(self.slippage_multipliers.get(symbol, 0)))
            if mult and mult > 0:
                if side == OrderSide.BUY:
                    execution_price = execution_price * (Decimal("1") + mult)
                else:
                    execution_price = execution_price * (Decimal("1") - mult)
        except Exception:
            pass
        return execution_price

    def _validate_exchange_rules(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        product: Product,
    ) -> tuple[Decimal, Decimal | None]:
        validator_price: Decimal | None
        if order_type == OrderType.MARKET:
            validator_price = None
        else:
            candidate = price if price is not None else effective_price
            validator_price = Decimal(str(candidate)) if candidate is not None else None

        vr = spec_validate_order(
            product=product,
            side=side.value,
            quantity=order_quantity,
            order_type=order_type.value.lower(),
            price=validator_price,
        )
        if not vr.ok:
            reason_code = vr.reason or "spec_violation"
            self._record_rejection(
                symbol, side.value, order_quantity, price or effective_price, reason_code
            )
            raise ValidationError(f"Spec validation failed: {reason_code}")

        if order_type == OrderType.LIMIT and price is not None:
            price = quantize_price_side_aware(
                Decimal(str(price)), product.price_increment, side.value
            )
        if vr.adjusted_price is not None:
            price = vr.adjusted_price
        if vr.adjusted_quantity is not None:
            order_quantity = vr.adjusted_quantity
        return order_quantity, price

    def _ensure_mark_is_fresh(self, symbol: str) -> None:
        try:
            if self.risk_manager.check_mark_staleness(symbol):
                raise ValidationError(f"Mark price is stale for {symbol}; halting order placement")
        except ValidationError:
            raise
        except Exception:
            pass

    def _enforce_slippage_guard(
        self,
        symbol: str,
        side: OrderSide,
        order_quantity: Decimal,
        effective_price: Decimal,
    ) -> None:
        try:
            snapshot = None
            if hasattr(self.broker, "get_market_snapshot"):
                snapshot = self.broker.get_market_snapshot(symbol)  # type: ignore[attr-defined]
            if snapshot:
                spread_bps = Decimal(str(snapshot.get("spread_bps", 0)))
                depth_l1 = Decimal(str(snapshot.get("depth_l1", 0)))
                notional = order_quantity * Decimal(str(effective_price))
                depth = depth_l1 if depth_l1 and depth_l1 > 0 else Decimal("1")
                impact_bps = Decimal("10000") * (notional / depth) * Decimal("0.5")
                expected_bps = spread_bps + impact_bps
                guard_limit = Decimal(str(self.risk_manager.config.slippage_guard_bps))
                if expected_bps > guard_limit:
                    raise ValidationError(
                        f"Expected slippage {expected_bps:.0f} bps exceeds guard {guard_limit}"
                    )
        except ValidationError:
            raise
        except Exception:
            pass

    def _run_pre_trade_validation(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_quantity: Decimal,
        effective_price: Decimal,
        product: Product,
        equity: Decimal,
        current_positions: dict[str, dict[str, Any]],
    ) -> None:
        self.risk_manager.pre_trade_validate(
            symbol=symbol,
            side=side.value,
            quantity=order_quantity,
            price=effective_price,
            product=product,
            equity=equity,
            current_positions=current_positions,
        )

    def _maybe_preview_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        effective_price: Decimal,
        stop_price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
    ) -> None:
        if not self.enable_order_preview:
            return
        try:
            tif_value = tif if isinstance(tif, TimeInForce) else (tif or TimeInForce.GTC)
            preview_data = self.broker.preview_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=order_quantity,
                price=effective_price,
                stop_price=stop_price,
                tif=tif_value,
                reduce_only=reduce_only,
                leverage=leverage,
            )
            self._record_preview(
                symbol, side, order_type, order_quantity, effective_price, preview_data
            )
        except ValidationError:
            raise
        except Exception as exc:
            logger.debug(f"Preview call failed: {exc}")

    def _finalize_reduce_only_flag(self, reduce_only: bool, symbol: str) -> bool:
        if self.risk_manager.is_reduce_only_mode():
            logger.info(f"Reduce-only mode active - forcing reduce_only=True for {symbol}")
            return True
        return reduce_only

    def _submit_order(
        self,
        *,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        order_quantity: Decimal,
        price: Decimal | None,
        effective_price: Decimal,
        stop_price: Decimal | None,
        tif: Any | None,
        reduce_only: bool,
        leverage: int | None,
        client_order_id: str | None,
    ) -> str | None:
        submit_id = (
            client_order_id or f"{self.bot_id}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        )
        try:
            get_logger().log_order_submission(
                client_order_id=submit_id,
                symbol=symbol,
                side=side.value,
                order_type=order_type.value,
                quantity=float(order_quantity),
                price=float(price) if price is not None else None,
            )
        except Exception:
            pass

        order = self.broker.place_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=order_quantity,
            price=price,
            stop_price=stop_price,
            tif=tif if tif is not None else None,
            reduce_only=reduce_only,
            leverage=leverage,
            client_id=submit_id,
        )

        if order and order.id:
            self.open_orders.append(order.id)
            display_price = price if price is not None else "market"
            logger.info(
                f"Order placed: {order.id} - {side.value} {order_quantity} {symbol} @ "
                f"{display_price} (reduce_only={reduce_only})"
            )
            logger.info(
                f"Trade recorded: {order.id} {side.value} {order_quantity} {symbol} @ "
                f"{display_price} (reduce_only={reduce_only})"
            )
            try:
                get_logger().log_order_status_change(
                    order_id=str(order.id),
                    client_order_id=getattr(order, "client_order_id", submit_id),
                    from_status=None,
                    to_status=getattr(order, "status", "SUBMITTED"),
                )
            except Exception:
                pass
            try:
                trade_quantity = getattr(order, "quantity", order_quantity)
                trade_payload = {
                    "order_id": order.id,
                    "client_order_id": getattr(order, "client_order_id", submit_id),
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(trade_quantity),
                    "price": str(order.price or price or effective_price or "market"),
                    "status": getattr(order, "status", "SUBMITTED"),
                }
                self.event_store.append_trade(self.bot_id, trade_payload)
            except Exception:
                pass
            return order.id

        logger.error(f"Order placement failed for {symbol}")
        try:
            self.event_store.append_error(
                bot_id=self.bot_id,
                message="order_placement_failed",
                context={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(order_quantity),
                },
            )
        except Exception:
            pass
        return None
