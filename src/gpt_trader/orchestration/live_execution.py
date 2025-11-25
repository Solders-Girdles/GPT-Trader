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

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, cast

from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.features.brokerages.coinbase.specs import validate_order as spec_validate_order
from gpt_trader.features.brokerages.core.interfaces import (
    IBrokerage,
    OrderSide,
    OrderType,
    Product,
)
from gpt_trader.features.live_trade.risk import LiveRiskManager, ValidationError
from gpt_trader.orchestration.execution import (
    GuardManager,
    OrderSubmitter,
    OrderValidator,
    RuntimeGuardState,
    StateCollector,
)
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_execution")

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
        settings: RuntimeSettings | None = None,
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

        provided_manager = risk_manager is not None
        store = event_store
        if store is None and provided_manager:
            store = getattr(risk_manager, "event_store", None)  # type: ignore[attr-defined]
        if store is None:
            store = EventStore()
        self.event_store = store

        self.risk_manager: LiveRiskManager
        if risk_manager is not None:
            self.risk_manager = risk_manager
            if getattr(self.risk_manager, "event_store", None) is not store:
                # Ensure the injected manager and its collaborators use the shared store.
                if hasattr(self.risk_manager, "set_event_store"):
                    self.risk_manager.set_event_store(store)
                else:  # Fallback for legacy managers without helper
                    setattr(self.risk_manager, "event_store", store)
        else:
            self.risk_manager = LiveRiskManager(event_store=store)
        self.bot_id = bot_id
        self.slippage_multipliers = slippage_multipliers or {}

        runtime_settings = settings or load_runtime_settings()
        # self._production_logger removed as it's now in StateCollector
        raw_integration = runtime_settings.raw_env.get("INTEGRATION_TEST_MODE", "")
        self._integration_mode = str(raw_integration).lower() in {"1", "true", "yes"}

        # Determine order preview setting
        preview_env = runtime_settings.raw_env.get("ORDER_PREVIEW_ENABLED")
        if enable_preview is not None:
            self.enable_order_preview = enable_preview
        elif preview_env is not None:
            self.enable_order_preview = preview_env.lower() in ("1", "true", "yes")
        else:
            self.enable_order_preview = False

        # Track open orders for cancellation on risk trips
        self.open_orders: list[str] = []
        # Track open orders for cancellation on risk trips
        self.open_orders: list[str] = []

        # Initialize helper modules
        self.state_collector: StateCollector = StateCollector(broker, settings=runtime_settings)
        self.order_submitter: OrderSubmitter = OrderSubmitter(
            broker,
            self.event_store,
            bot_id,
            self.open_orders,
            integration_mode=self._integration_mode,
        )
        self.order_validator: OrderValidator = OrderValidator(
            broker,
            self.risk_manager,
            self.enable_order_preview,
            self.order_submitter.record_preview,
            self.order_submitter.record_rejection,
        )
        self.guard_manager: GuardManager = GuardManager(
            broker,
            self.risk_manager,
            self.state_collector.calculate_equity_from_balances,
            self.open_orders,
            lambda: None,  # Cache invalidation handled by guard_manager itself
        )

        logger.info(
            "LiveExecutionEngine initialised",
            bot_id=bot_id,
            operation="live_execution_init",
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

            self.state_collector.log_collateral_update(
                collateral_balances, equity, collateral_total, balances
            )

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
            return cast(
                str | None,
                self.order_submitter.submit_order(
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
                ),
            )

        except ValidationError as exc:
            logger.warning(
                "Risk validation failed: %s",
                exc,
                operation="order_place",
                stage="validation",
                symbol=symbol,
                side=side.value,
            )
            rejection_price = price_decimal if price_decimal is not None else effective_price
            self.order_submitter.record_rejection(
                symbol, side.value, order_quantity, rejection_price, str(exc)
            )
            raise

        except Exception as exc:
            logger.error(
                "Order placement error: %s",
                exc,
                operation="order_place",
                stage="exception",
                symbol=symbol,
                side=side.value,
            )
            try:
                self.event_store.append_error(
                    bot_id=self.bot_id,
                    message="order_exception",
                    context={
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": str(order_quantity),
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
            raise

        finally:
            self.guard_manager.invalidate_cache()

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders (used on risk trips).

        Returns:
            Number of orders cancelled
        """
        return self.guard_manager.cancel_all_orders()

    def run_runtime_guards(self) -> None:
        """
        Run runtime risk guards and take action if needed.

        Should be called periodically (e.g., every minute).
        """
        self.guard_manager.safe_run_runtime_guards()

    def reset_daily_tracking(self) -> None:
        """Reset daily PnL tracking (call at start of trading day)."""
        try:
            # Collect fresh equity
            balances = self.broker.list_balances()
            equity, _, _ = self.state_collector.calculate_equity_from_balances(balances)

            # Reset risk manager daily tracking
            self.risk_manager.reset_daily_tracking(equity)
            logger.info(
                "Daily tracking reset",
                operation="daily_tracking",
                stage="reset",
                equity=float(equity),
            )

            # Invalidate guard cache
            self.guard_manager.invalidate_cache()

        except Exception as exc:
            logger.error(
                "Failed to reset daily tracking: %s",
                exc,
                operation="daily_tracking",
                stage="reset",
            )

    def _invalidate_runtime_guard_cache(self) -> None:
        """Backward compatibility wrapper for invalidate_cache."""
        self.guard_manager.invalidate_cache()
