"""Service for translating decisions into orders and placing them.

Responsibilities:
- Translate Decision DTO to order parameters
- Handle differences between AdvancedExecutionEngine and LiveExecutionEngine
- Manage order lock
- Track order statistics
- Update orders store
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.utilities.quantities import quantity_from

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
    from bot_v2.orchestration.live_execution import LiveExecutionEngine
    from bot_v2.persistence.orders_store import OrdersStore

logger = logging.getLogger(__name__)


class OrderPlacementService:
    """Service for placing orders based on trading decisions."""

    def __init__(
        self,
        orders_store: OrdersStore,
        order_stats: dict[str, int],
        broker: Any = None,
        dry_run: bool = False,
        *,
        metrics_server: Any | None = None,
        guardrails: Any | None = None,
        profile: str = "default",
    ) -> None:
        """Initialize order placement service.

        Args:
            orders_store: Store for persisting orders
            order_stats: Dict tracking attempted/successful/failed counts
            broker: Broker instance (needed for LiveExecutionEngine)
            dry_run: If True, log decisions but don't place orders
        """
        self._orders_store = orders_store
        self._order_stats = order_stats
        self._broker = broker
        self._dry_run = dry_run
        self._order_lock: asyncio.Lock | None = None
        self._metrics_server = metrics_server
        self._guardrails = guardrails
        self._profile = profile

    def _ensure_order_lock(self) -> asyncio.Lock:
        """Ensure order lock is initialized."""
        if self._order_lock is None:
            try:
                self._order_lock = asyncio.Lock()
            except RuntimeError as exc:
                logger.error("Unable to initialize async order lock: %s", exc)
                raise
        return self._order_lock

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
        exec_engine: AdvancedExecutionEngine | LiveExecutionEngine,
        reduce_only_mode: bool = False,
        default_time_in_force: str | None = None,
    ) -> None:
        """Execute a trading decision by placing an order.

        Args:
            symbol: Trading symbol
            decision: Decision object with action, quantity, etc.
            mark: Current mark price
            product: Product metadata
            position_state: Current position state (optional)
            exec_engine: Execution engine to use
            reduce_only_mode: If True, force reduce_only on all orders
            default_time_in_force: Default TIF if not specified in decision

        Raises:
            AssertionError: If inputs are invalid
        """
        # Validate inputs
        assert product is not None, "Missing product metadata"
        assert mark is not None and mark > 0, f"Invalid mark: {mark}"
        assert (
            position_state is None or "quantity" in position_state
        ), "Position state missing quantity"

        if self._dry_run:
            logger.info(f"DRY RUN: Would execute {decision.action.value} for {symbol}")
            return

        # Calculate order quantity
        position_quantity = quantity_from(position_state)
        if position_quantity is None:
            position_quantity = Decimal("0")

        if decision.action == Action.CLOSE:
            if not position_state or position_quantity == 0:
                logger.warning(f"No position to close for {symbol}")
                return
            order_quantity = abs(position_quantity)
        else:
            target_notional = getattr(decision, "target_notional", None)
            order_quantity: Decimal | None = None

            if target_notional not in (None, "", 0):
                try:
                    order_quantity = Decimal(str(target_notional)) / mark
                except (InvalidOperation, TypeError, ValueError) as exc:
                    logger.debug(
                        "Invalid target_notional %s for %s: %s",
                        target_notional,
                        symbol,
                        exc,
                    )
                    order_quantity = None

            if order_quantity is None:
                raw_quantity = getattr(decision, "quantity", None)
                if raw_quantity in (None, ""):
                    logger.warning(f"No quantity or notional in decision for {symbol}")
                    return
                try:
                    order_quantity = Decimal(str(raw_quantity))
                except (InvalidOperation, TypeError, ValueError) as exc:
                    logger.warning("Invalid quantity %s for %s: %s", raw_quantity, symbol, exc)
                    raise ValidationError(f"Invalid quantity for {symbol}") from exc

        # Determine order side
        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
        if decision.action == Action.CLOSE:
            side = (
                OrderSide.SELL
                if position_state and position_state.get("side") == "long"
                else OrderSide.BUY
            )

        # Determine reduce_only
        reduce_only = decision.reduce_only or reduce_only_mode or decision.action == Action.CLOSE

        # Extract order parameters
        order_type = getattr(decision, "order_type", OrderType.MARKET)
        limit_price = getattr(decision, "limit_price", None)
        stop_price = getattr(decision, "stop_trigger", None)
        tif = getattr(decision, "time_in_force", None)

        # Parse time_in_force
        try:
            if isinstance(tif, str):
                tif = TimeInForce[tif.upper()]
            elif tif is None and isinstance(default_time_in_force, str):
                tif = TimeInForce[default_time_in_force.upper()]
        except Exception:
            tif = None

        # Normalize order type
        normalised_order_type = (
            order_type
            if isinstance(order_type, OrderType)
            else (
                OrderType[order_type.upper()] if isinstance(order_type, str) else OrderType.MARKET
            )
        )

        # Build kwargs based on engine type
        place_kwargs = self._build_place_kwargs(
            exec_engine=exec_engine,
            symbol=symbol,
            side=side,
            quantity=order_quantity,
            order_type=normalised_order_type,
            reduce_only=reduce_only,
            leverage=decision.leverage,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            product=product,
        )

        # Guard rail checks (e.g., dry-run enforcement, caps)
        if self._guardrails is not None:
            guard_context = {
                "symbol": symbol,
                "decision": decision,
                "mark": mark,
                "product": product,
                "order_kwargs": place_kwargs,
                "profile": self._profile,
            }
            guard_result = self._guardrails.check_order(guard_context)
            if not guard_result.allowed:
                guard_name = guard_result.guard or "unknown"
                reason = guard_result.reason or "blocked"
                logger.info("Order blocked by guard %s: %s", guard_name, reason)

                self._order_stats.setdefault("attempted", 0)
                self._order_stats.setdefault("successful", 0)
                self._order_stats.setdefault("failed", 0)
                self._order_stats["attempted"] += 1

                if guard_name == "dry_run":
                    self._order_stats["successful"] += 1
                else:
                    self._order_stats["failed"] += 1

                if self._metrics_server:
                    self._metrics_server.record_order_attempt(
                        "attempted", profile=self._profile
                    )
                    self._metrics_server.record_order_attempt(
                        "success" if guard_name == "dry_run" else "failed",
                        profile=self._profile,
                    )
                    self._metrics_server.record_guard_trip(
                        guard_name, reason, profile=self._profile
                    )
                    if self._guardrails:
                        self._metrics_server.update_error_streak(
                            self._guardrails.get_error_streak(), profile=self._profile
                        )
                if self._guardrails:
                    self._guardrails.record_success()
                return

        # Place order
        try:
            order = await self._place_order(exec_engine, **place_kwargs)
            if order:
                logger.info(f"Order placed successfully: {order.id}")
                self._handle_guard_success()
            else:
                logger.warning(f"Order rejected or failed for {symbol}")
                self._handle_guard_error("order_failed")
        except (ValidationError, ExecutionError, RiskValidationError):
            # Propagate expected validation/execution failures to caller
            self._handle_guard_error("order_validation_error")
            raise
        except Exception as exc:
            logger.error("Error executing decision for %s: %s", symbol, exc, exc_info=True)
            self._handle_guard_error("order_exception")

    def _build_place_kwargs(
        self,
        exec_engine: Any,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        order_type: OrderType,
        reduce_only: bool,
        leverage: Any,
        limit_price: Any,
        stop_price: Any,
        tif: TimeInForce | None,
        product: Product,
    ) -> dict[str, Any]:
        """Build kwargs dict for place_order call.

        AdvancedExecutionEngine and LiveExecutionEngine have different signatures.
        """
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        base_kwargs: dict[str, Any] = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "reduce_only": reduce_only,
            "leverage": leverage,
        }

        if isinstance(exec_engine, AdvancedExecutionEngine):
            base_kwargs.update(
                {
                    "limit_price": limit_price,
                    "stop_price": stop_price,
                    "time_in_force": tif or TimeInForce.GTC,
                }
            )
        else:
            base_kwargs.update(
                {
                    "product": product,
                    "price": limit_price,
                    "stop_price": stop_price,
                    "tif": tif or None,
                }
            )

        return base_kwargs

    async def _place_order(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        """Place order with locking and error handling."""
        try:
            lock = self._ensure_order_lock()
            async with lock:
                return await self._place_order_inner(exec_engine, **kwargs)
        except (ValidationError, RiskValidationError, ExecutionError) as e:
            logger.warning(f"Order validation/execution failed: {e}")
            self._order_stats["failed"] += 1
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {e}", exc_info=True)
            self._order_stats["failed"] += 1
            return None

    async def _place_order_inner(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        """Inner order placement logic."""
        from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

        self._order_stats["attempted"] += 1

        def _place() -> Any:
            return exec_engine.place_order(**kwargs)

        result = await asyncio.to_thread(_place)

        # Handle result based on engine type
        if isinstance(exec_engine, AdvancedExecutionEngine):
            order = result
        else:
            order = None
            if result and self._broker:
                order = await asyncio.to_thread(self._broker.get_order, result)

        if order:
            self._orders_store.upsert(order)
            self._order_stats["successful"] += 1
            order_quantity = quantity_from(order)
            # Normalize side to handle both enum and string
            side_str = getattr(order.side, "value", order.side)
            logger.info(f"Order recorded: {order.id} {side_str} {order_quantity} {order.symbol}")
            self._handle_guard_success()
            return order

        self._order_stats["failed"] += 1
        logger.warning("Order attempt failed (no order returned)")
        self._handle_guard_error("order_no_result")
        return None

    def _handle_guard_error(self, reason: str) -> None:
        if self._guardrails is None:
            return

        streak, triggered = self._guardrails.record_error(reason)
        if self._metrics_server:
            self._metrics_server.update_error_streak(streak, profile=self._profile)
            if triggered:
                self._metrics_server.record_guard_trip(
                    "circuit_breaker", reason, profile=self._profile
                )

    def _handle_guard_success(self) -> None:
        if self._guardrails is None:
            return
        self._guardrails.record_success()
        if self._metrics_server:
            self._metrics_server.update_error_streak(
                self._guardrails.get_error_streak(), profile=self._profile
            )
