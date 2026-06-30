"""Trading-engine guard-stack and user-event initialization.

Construction-time wiring for the live TradingEngine: builds the StateCollector,
OrderSubmitter, OrderValidator, and GuardManager pre-trade guard stack, and wires
the Coinbase user-event handler. Extracted from strategy.py (which calls these
during __init__) following the engine's existing collaborator-function pattern;
the engine instance is passed in and its attributes are populated in place.
"""

from __future__ import annotations

from typing import Any

from gpt_trader.features.live_trade.execution.guard_manager import GuardManager
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.features.live_trade.execution.state_collection import StateCollector
from gpt_trader.features.live_trade.execution.validation import OrderValidator
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="trading_engine")


def init_guard_stack(engine: Any) -> None:
    """Initialize StateCollector, OrderValidator, OrderSubmitter for pre-trade guards."""
    # Event store fallback
    event_store = engine.context.event_store or EventStore()
    engine._event_store = event_store
    bot_id = str(engine.context.bot_id or engine.context.config.profile or "live")

    # Orders store for durable restart (optional)
    orders_store = engine.context.orders_store
    if orders_store is None and engine.context.container is not None:
        orders_store = getattr(engine.context.container, "orders_store", None)
    if orders_store is not None:
        try:
            orders_store.initialize()
        except Exception as exc:
            logger.warning(
                "Failed to initialize orders store",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="orders_store_init",
            )
            orders_store = None
    engine._orders_store = orders_store

    # Broker and risk manager must exist
    broker = engine.context.broker
    risk_manager = engine.context.risk_manager

    # Track open orders
    engine._open_orders = []
    engine._rehydrate_open_orders()

    # StateCollector: needs broker, config
    engine._state_collector = StateCollector(
        broker=broker,  # type: ignore[arg-type]
        config=engine.context.config,
        integration_mode=False,
    )

    # OrderSubmitter: broker + event store + bot_id + open_orders
    engine._order_submitter = OrderSubmitter(
        broker=broker,  # type: ignore[arg-type]
        event_store=event_store,
        bot_id=bot_id,
        open_orders=engine._open_orders,
        enable_retries=getattr(engine.context.config, "order_submission_retries_enabled", False),
        orders_store=engine._orders_store,
        integration_mode=False,
    )

    # Failure tracker from container (not global) with escalation callback
    container = engine.context.container
    if container is None:
        raise RuntimeError(
            "TradingEngine requires a container in context. "
            "Pass container=ApplicationContainer(config) to CoordinatorContext."
        )
    failure_tracker = container.validation_failure_tracker

    # Wire escalation callback: on repeated validation failures, pause + reduce-only
    def _on_validation_escalation() -> None:
        """Handle validation infrastructure failures by pausing and setting reduce-only."""
        if risk_manager is None:
            return

        risk_manager.set_reduce_only_mode(True, reason="validation_failures")
        cooldown = 180
        if risk_manager.config is not None:
            cooldown = risk_manager.config.validation_failure_cooldown_seconds
        engine._degradation.pause_all(
            seconds=cooldown,
            reason="validation_failures",
            allow_reduce_only=True,
        )
        logger.warning(
            "Validation escalation triggered - pausing trading",
            cooldown_seconds=cooldown,
            operation="degradation",
            stage="validation_escalation",
        )

    failure_tracker.escalation_callback = _on_validation_escalation

    # OrderValidator: broker + risk_manager + preview config + callbacks + tracker
    engine._order_validator = None
    if risk_manager is not None:
        engine._order_validator = OrderValidator(
            broker=broker,  # type: ignore[arg-type]
            risk_manager=risk_manager,
            enable_order_preview=engine.context.config.enable_order_preview,
            record_preview_callback=engine._order_submitter.record_preview,
            record_rejection_callback=engine._order_submitter.record_rejection,
            failure_tracker=failure_tracker,
            broker_calls=engine._broker_calls,
        )

    # GuardManager: runtime guards (daily loss, liquidation buffer, volatility)
    engine._guard_manager = None
    if broker is not None and risk_manager is not None:
        engine._guard_manager = GuardManager(
            broker=broker,  # type: ignore[arg-type]
            risk_manager=risk_manager,
            equity_calculator=engine._state_collector.calculate_equity_from_balances,
            open_orders=engine._open_orders,
            invalidate_cache_callback=lambda: None,
            cancel_retries_enabled=getattr(
                engine.context.config, "order_submission_retries_enabled", False
            ),
        )


def init_user_event_handler(engine: Any) -> None:
    """Initialize Coinbase WS user-event handling for live order updates."""
    if getattr(engine.context.config, "dry_run", False):
        logger.info(
            "Dry-run enabled; skipping Coinbase user-event handling",
            operation="user_events",
            stage="skip",
        )
        return

    broker = engine.context.broker
    if broker is None:
        return

    module_name = getattr(broker, "__module__", "")
    if "coinbase" not in module_name:
        return

    from gpt_trader.features.brokerages.coinbase.user_event_handler import (
        CoinbaseUserEventHandler,
    )

    market_data_service = None
    product_catalog = None
    container = engine.context.container
    if container is not None:
        market_data_service = getattr(container, "market_data_service", None)
        product_catalog = getattr(container, "product_catalog", None)

    engine._user_event_handler = CoinbaseUserEventHandler(
        broker=broker,
        orders_store=engine._orders_store,
        event_store=engine.context.event_store,
        bot_id=str(engine.context.bot_id or engine.context.config.profile or "live"),
        market_data_service=market_data_service,
        symbols=list(engine.context.config.symbols),
        product_catalog=product_catalog,
    )
