"""
Live execution engine with risk management integration.

Phase 5: Risk engine integration for perpetuals.

This module has been refactored to delegate to focused helper modules:
- execution.guards: Runtime guard management
- execution.validation: Pre-trade validation
- execution.order_submission: Order submission and recording
- execution.state_collection: Account state collection
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from bot_v2.features.brokerages.coinbase.specs import validate_order as spec_validate_order
from bot_v2.features.brokerages.core.interfaces import (
    Balance,
    IBrokerage,
    OrderSide,
    OrderType,
    Product,
)
from bot_v2.features.live_trade.guard_errors import RiskGuardError
from bot_v2.features.live_trade.risk import LiveRiskManager, ValidationError
from bot_v2.monitoring.system import get_logger
from bot_v2.orchestration.execution import (
    GuardManager,
    OrderSubmitter,
    OrderValidator,
    RuntimeGuardState,
    StateCollector,
)
from bot_v2.persistence.event_store import EventStore

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
__all__ = ["LiveExecutionEngine", "LiveOrder", "RuntimeGuardState", "spec_validate_order"]


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


class LiveExecutionEngine:
    """Live execution with integrated risk controls for perpetuals.

    Enforces risk checks before order placement and monitors runtime guards.

    This class has been refactored to delegate to helper modules while
    maintaining backward compatibility.
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
            slippage_multipliers: Symbol-specific slippage multipliers
            enable_preview: Enable order preview (defaults to env var)
        """
        self.broker = broker
        self.risk_manager = risk_manager or LiveRiskManager()
        self.event_store = event_store or EventStore()
        self.bot_id = bot_id
        self.slippage_multipliers = slippage_multipliers or {}

        # Determine order preview setting
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

        # Initialize helper modules
        self.state_collector = StateCollector(broker)
        self.order_submitter = OrderSubmitter(broker, event_store, bot_id, self.open_orders)
        self.order_validator = OrderValidator(
            broker,
            risk_manager,
            self.enable_order_preview,
            self.order_submitter.record_preview,
            self.order_submitter.record_rejection,
        )
        self.guard_manager = GuardManager(
            broker,
            risk_manager,
            self.state_collector.calculate_equity_from_balances,
            self.cancel_all_orders,
            lambda: None,  # Cache invalidation handled by guard_manager itself
        )

        logger.info(f"LiveExecutionEngine initialized for {bot_id}")

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
            stop_price: Stop price for stop orders
            tif: Time in force
            reduce_only: Force reduce-only order
            leverage: Target leverage for perpetuals
            product: Product metadata (fetched if None)
            client_order_id: Client order ID

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
            # Ensure product is available
            product = self.state_collector.require_product(symbol, product)

            # Collect account state
            (
                balances,
                equity,
                collateral_balances,
                collateral_total,
                current_positions,
            ) = self.state_collector.collect_account_state()

            self._log_collateral_update(collateral_balances, equity, collateral_total, balances)

            # Build positions dict for validation
            current_positions_dict = self.state_collector.build_positions_dict(current_positions)

            # Resolve effective price
            effective_price = self.state_collector.resolve_effective_price(
                symbol, side.value, price_decimal, product
            )

            # Validate exchange rules and quantization
            order_quantity, price_decimal = self.order_validator.validate_exchange_rules(
                symbol,
                side,
                order_type,
                order_quantity,
                price_decimal,
                effective_price,
                product,
            )

            # Check mark price freshness
            self.order_validator.ensure_mark_is_fresh(symbol)

            # Enforce slippage guard
            self.order_validator.enforce_slippage_guard(
                symbol, side, order_quantity, effective_price
            )

            # Run comprehensive pre-trade validation
            self.order_validator.run_pre_trade_validation(
                symbol=symbol,
                side=side,
                order_quantity=order_quantity,
                effective_price=effective_price,
                product=product,
                equity=equity,
                current_positions=current_positions_dict,
            )

            # Preview order if enabled
            self.order_validator.maybe_preview_order(
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

            # Finalize reduce-only flag
            is_reduce_only = self.order_validator.finalize_reduce_only_flag(reduce_only, symbol)

            # Submit order
            return self.order_submitter.submit_order(
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
            self.order_submitter.record_rejection(
                symbol, side.value, order_quantity, rejection_price, str(e)
            )
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
            self.guard_manager.invalidate_cache()

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
            self.guard_manager.invalidate_cache()

        return cancelled

    def run_runtime_guards(self) -> None:
        """
        Run runtime risk guards and take action if needed.

        Should be called periodically (e.g., every minute).
        """
        try:
            now = time.time()
            force_full = self.guard_manager.should_run_full_guard(now)
            self.guard_manager.run_runtime_guards(force_full=force_full)

            # Store state for potential incremental runs
            self.guard_manager._runtime_guard_last_run_ts = now

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
                self.guard_manager.invalidate_cache()

        except Exception as e:
            logger.error(f"Runtime guards error: {e}")

    def reset_daily_tracking(self) -> None:
        """Reset daily PnL tracking (call at start of trading day)."""
        try:
            # Collect fresh equity
            balances = self.broker.list_balances()
            equity, _, _ = self.state_collector.calculate_equity_from_balances(balances)

            # Reset risk manager daily tracking
            self.risk_manager.reset_daily_tracking(equity)
            logger.info(f"Daily tracking reset with equity: {equity}")

            # Invalidate guard cache
            self.guard_manager.invalidate_cache()

        except Exception as e:
            logger.error(f"Failed to reset daily tracking: {e}")

    def _invalidate_runtime_guard_cache(self) -> None:
        """Backward compatibility wrapper for invalidate_cache."""
        self.guard_manager.invalidate_cache()

    def _log_collateral_update(
        self,
        collateral_balances: list[Balance],
        equity: Decimal,
        collateral_total: Decimal,
        all_balances: list[Balance],
    ) -> None:
        """Log collateral balance changes."""
        if not collateral_balances:
            return

        total_available = sum(b.available for b in collateral_balances)

        if self._last_collateral_available is not None:
            change = total_available - self._last_collateral_available
            if abs(change) > Decimal("0.01"):
                logger.info(
                    f"Collateral available changed: {self._last_collateral_available} → "
                    f"{total_available} (Δ {change:+.2f})"
                )

        self._last_collateral_available = total_available

        # Log to telemetry
        try:
            get_logger().log_balance_update(
                available=float(total_available),
                total=float(collateral_total),
                equity=float(equity),
            )
        except Exception:
            pass
