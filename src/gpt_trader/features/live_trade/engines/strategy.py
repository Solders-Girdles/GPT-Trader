"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.

State Recovery:
On startup, reads `price_tick` events from EventStore to restore price history.
During operation, persists price ticks to EventStore for crash recovery.

Streaming Lifecycle:
When enabled, starts WebSocket streaming for real-time market data.
Includes WS health watchdog that monitors staleness and triggers degradation.
"""

import asyncio
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from importlib import metadata
from typing import Any

from gpt_trader.app.health_server import HealthState
from gpt_trader.config.constants import HEALTH_CHECK_INTERVAL_SECONDS
from gpt_trader.core import OrderSide, OrderType, Position, Product
from gpt_trader.features.live_trade.degradation import DegradationState
from gpt_trader.features.live_trade.engines.base import (
    BaseEngine,
    CoordinatorContext,
    HealthStatus,
)
from gpt_trader.features.live_trade.engines.equity_calculator import (
    EquityCalculator,
)
from gpt_trader.features.live_trade.engines.price_tick_store import (
    EVENT_PRICE_TICK,
    PriceTickStore,
)
from gpt_trader.features.live_trade.engines.system_maintenance import (
    SystemMaintenanceService,
)
from gpt_trader.features.live_trade.engines.telemetry_health import (
    extract_mark_from_message,
    update_mark_and_metrics,
)
from gpt_trader.features.live_trade.engines.telemetry_streaming import (
    _handle_stream_task_completion,
    _run_stream_loop,
    _run_stream_loop_async,
    _schedule_coroutine,
    _should_enable_streaming,
    _start_streaming,
    _stop_streaming,
    start_streaming_background,
    stop_streaming_background,
)
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor
from gpt_trader.features.live_trade.execution.decision_trace import OrderDecisionTrace
from gpt_trader.features.live_trade.execution.guard_manager import GuardManager
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.features.live_trade.execution.state_collection import StateCollector
from gpt_trader.features.live_trade.execution.submission_result import (
    OrderSubmissionResult,
    OrderSubmissionStatus,
)
from gpt_trader.features.live_trade.execution.validation import OrderValidator
from gpt_trader.features.live_trade.factory import create_strategy
from gpt_trader.features.live_trade.guard_errors import GuardError
from gpt_trader.features.live_trade.lifecycle import (
    ENGINE_TRANSITIONS,
    EngineState,
    LifecycleStateMachine,
)
from gpt_trader.features.live_trade.risk.manager import ValidationError
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    Action,
    Decision,
)
from gpt_trader.logging.correlation import correlation_context
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.health_checks import HealthCheckRunner
from gpt_trader.monitoring.heartbeat import HeartbeatService
from gpt_trader.monitoring.metrics_collector import record_histogram
from gpt_trader.monitoring.profiling import profile_span
from gpt_trader.monitoring.status_reporter import StatusReporter
from gpt_trader.observability.tracing import trace_span
from gpt_trader.persistence.event_store import EventStore
from gpt_trader.persistence.orders_store import (
    OrderRecord,
)
from gpt_trader.persistence.orders_store import (
    OrderStatus as PersistedOrderStatus,
)
from gpt_trader.utilities.async_tools import BoundedToThread
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="trading_engine")

# Consecutive order reconciliation drift detections required before triggering
# an escalation (pause + reduce-only). Keep this conservative to avoid
# flapping on transient list_orders inconsistencies.
ORDER_RECONCILIATION_DRIFT_MAX_FAILURES = 3

# Re-export for backward compatibility
__all__ = ["TradingEngine", "EVENT_PRICE_TICK"]


class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.

    Supports state recovery via EventStore persistence.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self._lifecycle: LifecycleStateMachine[EngineState] = LifecycleStateMachine(
            initial_state=EngineState.INIT,
            entity="trading_engine",
            transitions=ENGINE_TRANSITIONS,
            logger=logger,
        )
        # Create strategy via factory (supports baseline and mean_reversion)
        self.strategy = create_strategy(self.context.config)
        self._current_positions: dict[str, Position] = {}
        self._rehydrated = False
        self._cycle_count = 0

        # Initialize price tick store for state recovery
        self._price_tick_store = PriceTickStore(
            event_store=context.event_store,
            symbols=list(context.config.symbols),
            bot_id=context.bot_id,
        )

        # Initialize heartbeat service
        self._heartbeat = HeartbeatService(
            event_store=context.event_store,
            ping_url=getattr(context.config, "heartbeat_url", None),
            interval_seconds=getattr(context.config, "heartbeat_interval", 60),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "heartbeat_enabled", True),
        )

        # Initialize status reporter
        self._status_reporter = StatusReporter(
            status_file=getattr(context.config, "status_file", "status.json"),
            file_write_interval=getattr(context.config, "status_interval", 60),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "status_enabled", True),
        )
        self._status_reporter.set_heartbeat_service(self._heartbeat)

        # Initialize system maintenance service (health reporting + pruning)
        self._system_maintenance = SystemMaintenanceService(
            status_reporter=self._status_reporter,
            event_store=context.event_store,
        )

        # System health tracking
        self._last_latency = 0.0
        self._connection_status = "UNKNOWN"

        # Initialize graceful degradation state
        self._degradation = DegradationState()
        self._unfilled_order_alerts: dict[str, float] = {}
        self._order_reconciliation_drift_failures = 0
        self._order_reconciliation_drift_escalated = False

        broker_calls = getattr(context, "broker_calls", None)
        if broker_calls is not None and not asyncio.iscoroutinefunction(
            getattr(broker_calls, "__call__", None)
        ):
            broker_calls = None
        if broker_calls is None:
            broker_call_limit = getattr(context.config, "max_concurrent_broker_calls", None)
            if broker_call_limit is None:
                broker_call_limit = getattr(context.config, "max_concurrent_rest_calls", 5)
            try:
                raw_limit = broker_call_limit if broker_call_limit is not None else 5
                broker_call_limit = int(raw_limit)
            except (TypeError, ValueError):
                broker_call_limit = 5
            broker_call_limit = max(1, broker_call_limit)
            use_dedicated_executor = (
                getattr(context.config, "broker_calls_use_dedicated_executor", False) is True
            )
            broker_calls = BoundedToThread(
                max_concurrency=broker_call_limit,
                use_dedicated_executor=use_dedicated_executor,
            )

        self._broker_calls = broker_calls

        # Initialize equity calculator (extracted for reusability)
        self._equity_calculator = EquityCalculator(
            config=context.config,
            degradation=self._degradation,
            risk_manager=context.risk_manager,
            price_history=self._price_tick_store.price_history,
            broker_calls=self._broker_calls,
        )

        # Initialize health check runner for active /health probes
        health_state = context.container.health_state if context.container else HealthState()
        self._health_check_runner = HealthCheckRunner(
            health_state=health_state,
            broker=context.broker,
            degradation_state=self._degradation,
            risk_manager=context.risk_manager,
            interval_seconds=HEALTH_CHECK_INTERVAL_SECONDS,
            message_stale_seconds=getattr(context.config, "ws_message_stale_seconds", 60.0),
            heartbeat_stale_seconds=getattr(context.config, "ws_heartbeat_stale_seconds", 120.0),
            broker_calls=self._broker_calls,
        )

        # Initialize streaming lifecycle attributes
        self._ws_stop: threading.Event | None = None
        self._pending_stream_config: tuple[list[str], int] | None = None
        self._stream_task: asyncio.Task[Any] | None = None
        self._loop_task_handle: asyncio.Task[Any] | None = None
        self._market_monitor: Any = None  # Market monitor for telemetry

        # WS health watchdog attributes
        self._ws_health_task: asyncio.Task[Any] | None = None
        self._ws_reconnect_attempts: int = 0
        self._ws_reconnect_delay: float = 1.0
        self._ws_last_health_check: float = 0.0

        # Initialize pre-trade guard stack (Option A: embedded guards)
        self._init_guard_stack()
        self._user_event_handler: Any | None = None
        self._init_user_event_handler()

    def _init_guard_stack(self) -> None:
        """Initialize StateCollector, OrderValidator, OrderSubmitter for pre-trade guards."""
        # Event store fallback
        event_store = self.context.event_store or EventStore()
        self._event_store = event_store
        bot_id = str(self.context.bot_id or self.context.config.profile or "live")

        # Orders store for durable restart (optional)
        orders_store = self.context.orders_store
        if orders_store is None and self.context.container is not None:
            orders_store = getattr(self.context.container, "orders_store", None)
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
        self._orders_store = orders_store

        # Broker and risk manager must exist
        broker = self.context.broker
        risk_manager = self.context.risk_manager

        # Track open orders
        self._open_orders: list[str] = []
        self._rehydrate_open_orders()

        # StateCollector: needs broker, config
        self._state_collector = StateCollector(
            broker=broker,  # type: ignore[arg-type]
            config=self.context.config,
            integration_mode=False,
        )

        # OrderSubmitter: broker + event store + bot_id + open_orders
        self._order_submitter = OrderSubmitter(
            broker=broker,  # type: ignore[arg-type]
            event_store=event_store,
            bot_id=bot_id,
            open_orders=self._open_orders,
            enable_retries=getattr(self.context.config, "order_submission_retries_enabled", False),
            orders_store=self._orders_store,
            integration_mode=False,
        )

        # Failure tracker from container (not global) with escalation callback
        container = self.context.container
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
            self._degradation.pause_all(
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
        self._order_validator: OrderValidator | None = None
        if risk_manager is not None:
            self._order_validator = OrderValidator(
                broker=broker,  # type: ignore[arg-type]
                risk_manager=risk_manager,
                enable_order_preview=self.context.config.enable_order_preview,
                record_preview_callback=self._order_submitter.record_preview,
                record_rejection_callback=self._order_submitter.record_rejection,
                failure_tracker=failure_tracker,
                broker_calls=self._broker_calls,
            )

        # GuardManager: runtime guards (daily loss, liquidation buffer, volatility)
        self._guard_manager: GuardManager | None = None
        if broker is not None and risk_manager is not None:
            self._guard_manager = GuardManager(
                broker=broker,  # type: ignore[arg-type]
                risk_manager=risk_manager,
                equity_calculator=self._state_collector.calculate_equity_from_balances,
                open_orders=self._open_orders,
                invalidate_cache_callback=lambda: None,
                cancel_retries_enabled=getattr(
                    self.context.config, "order_submission_retries_enabled", False
                ),
            )

    def _init_user_event_handler(self) -> None:
        """Initialize Coinbase WS user-event handling for live order updates."""
        if getattr(self.context.config, "dry_run", False):
            logger.info(
                "Dry-run enabled; skipping Coinbase user-event handling",
                operation="user_events",
                stage="skip",
            )
            return

        broker = self.context.broker
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
        container = self.context.container
        if container is not None:
            market_data_service = getattr(container, "market_data_service", None)
            product_catalog = getattr(container, "product_catalog", None)

        self._user_event_handler = CoinbaseUserEventHandler(
            broker=broker,
            orders_store=self._orders_store,
            event_store=self.context.event_store,
            bot_id=str(self.context.bot_id or self.context.config.profile or "live"),
            market_data_service=market_data_service,
            symbols=list(self.context.config.symbols),
            product_catalog=product_catalog,
        )

    def _rehydrate_open_orders(self) -> None:
        """Restore open order IDs from the orders store after a restart."""
        if self._orders_store is None:
            return
        bot_id = str(self.context.bot_id or self.context.config.profile or "")
        try:
            pending = self._orders_store.get_pending_orders(bot_id=bot_id or None)
        except Exception as exc:
            logger.warning(
                "Failed to rehydrate open orders",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="order_recovery",
            )
            return

        if not pending:
            return

        seen: set[str] = set()
        order_ids: list[str] = []
        for order in pending:
            if order.order_id in seen:
                continue
            seen.add(order.order_id)
            order_ids.append(order.order_id)

        self._open_orders[:] = order_ids
        logger.info(
            "Rehydrated open orders from persistence",
            count=len(order_ids),
            operation="order_recovery",
        )

    async def _recover_unknown_bot_orders(
        self,
        orders: list[Any],
        *,
        bot_id: str,
    ) -> list[OrderRecord]:
        if not orders or self._orders_store is None:
            return []
        now = datetime.now(timezone.utc)
        recovered: list[OrderRecord] = []
        for order in orders:
            record = self._build_record_from_broker_order(order, bot_id=bot_id, now=now)
            if record is None:
                continue
            metadata = dict(record.metadata or {})
            metadata["note"] = "backfilled_from_broker"
            record = replace(record, metadata=metadata)
            try:
                await self._broker_calls(self._orders_store.upsert_by_client_id, record)
            except Exception as exc:
                logger.warning(
                    "Failed to persist recovered order",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="order_reconciliation",
                    stage="persist_recovery",
                    order_id=record.order_id,
                    client_order_id=record.client_order_id,
                )
                continue
            if record.order_id not in self._open_orders:
                self._open_orders.append(record.order_id)
            recovered.append(record)

        if recovered:
            payload = {
                "timestamp": time.time(),
                "bot_id": bot_id,
                "recovered_order_ids": [record.order_id for record in recovered][:10],
                "recovered_order_count": len(recovered),
                "open_order_count": len(self._open_orders),
            }
            self._append_event("order_reconciliation_recovered", payload)
            logger.info(
                "Recovered unknown bot orders",
                recovered_order_count=len(recovered),
                operation="order_reconciliation",
                stage="recover",
            )

        return recovered

    async def _refresh_missing_persisted_orders(
        self,
        records: list[OrderRecord],
        *,
        bot_id: str,
    ) -> list[Any]:
        if not records or self._orders_store is None:
            return []
        broker = self.context.broker
        if broker is None:
            return []
        get_order = getattr(broker, "get_order", None)
        if not callable(get_order):
            return []
        now = datetime.now(timezone.utc)
        refreshed_orders: list[Any] = []
        for record in records:
            try:
                order = await self._broker_calls(get_order, record.order_id)
                if order is None and record.client_order_id != record.order_id:
                    order = await self._broker_calls(get_order, record.client_order_id)
            except Exception as exc:
                logger.warning(
                    "Failed to load order during reconciliation",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="order_reconciliation",
                    stage="load_missing",
                    order_id=record.order_id,
                    client_order_id=record.client_order_id,
                )
                continue
            if order is None:
                continue
            update_record = self._build_record_from_broker_order(order, bot_id=bot_id, now=now)
            if update_record is None:
                continue
            update_metadata = dict(update_record.metadata or {})
            update_metadata["note"] = "refresh_missing"
            metadata = self._merge_metadata(record.metadata, update_metadata)
            updated = replace(
                record,
                order_id=update_record.order_id,
                client_order_id=update_record.client_order_id,
                symbol=update_record.symbol or record.symbol,
                side=update_record.side or record.side,
                order_type=update_record.order_type or record.order_type,
                quantity=(
                    update_record.quantity
                    if update_record.quantity != Decimal("0")
                    else record.quantity
                ),
                price=update_record.price if update_record.price is not None else record.price,
                status=update_record.status,
                filled_quantity=(
                    update_record.filled_quantity
                    if update_record.filled_quantity != Decimal("0")
                    else record.filled_quantity
                ),
                average_fill_price=(update_record.average_fill_price or record.average_fill_price),
                updated_at=now,
                metadata=metadata,
                bot_id=record.bot_id or bot_id,
                time_in_force=update_record.time_in_force or record.time_in_force,
            )
            try:
                await self._broker_calls(self._orders_store.upsert_by_client_id, updated)
            except Exception as exc:
                logger.warning(
                    "Failed to persist order refresh",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="order_reconciliation",
                    stage="persist_refresh",
                    order_id=updated.order_id,
                    client_order_id=updated.client_order_id,
                )
                continue

            if updated.status in {
                PersistedOrderStatus.PENDING,
                PersistedOrderStatus.OPEN,
                PersistedOrderStatus.PARTIALLY_FILLED,
            }:
                if updated.order_id not in self._open_orders:
                    self._open_orders.append(updated.order_id)
                refreshed_orders.append(order)
            else:
                if updated.order_id in self._open_orders:
                    self._open_orders.remove(updated.order_id)

        return refreshed_orders

    async def _reconcile_open_orders(self, orders: list[Any]) -> None:
        """Reconcile internal open-order tracking with broker + persistence state.

        This addresses a crash-recovery edge case: orders are persisted as pending with
        ``order_id == client_order_id == submit_id`` before the broker responds with
        the canonical broker order ID. After restart, rehydration can restore the
        submit_id into ``self._open_orders``. On the next audit, we normalize tracked
        IDs back to broker order IDs when the broker returns both IDs.
        """
        bot_id = str(self.context.bot_id or self.context.config.profile or "live")
        prefix = f"{bot_id}_"

        pending: list[OrderRecord] = []
        pending_by_client_id: dict[str, OrderRecord] = {}
        tracked_ids: set[str] = {str(order_id) for order_id in self._open_orders}

        if self._orders_store is not None:
            try:
                pending = await self._broker_calls(
                    self._orders_store.get_pending_orders,
                    bot_id=bot_id or None,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load pending orders during audit",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    operation="order_reconciliation",
                    stage="load_pending",
                )
                pending = []

        for store_record in pending:
            pending_by_client_id[str(store_record.client_order_id)] = store_record
            tracked_ids.add(str(store_record.client_order_id))
            tracked_ids.add(str(store_record.order_id))

        if not orders:
            refreshed_orders: list[Any] = []
            if pending:
                refreshed_orders = await self._refresh_missing_persisted_orders(
                    pending,
                    bot_id=bot_id,
                )
            if refreshed_orders:
                orders = refreshed_orders
            else:
                if not pending and self._open_orders:
                    self._open_orders.clear()
                self._order_reconciliation_drift_failures = 0
                self._order_reconciliation_drift_escalated = False
                return

        order_index: dict[str, Any] = {}

        def _index_order(order: Any) -> None:
            if order is None:
                return
            order_id = self._get_order_field(order, "order_id", "id")
            client_order_id = self._get_order_field(order, "client_order_id", "client_id")
            if order_id is not None:
                order_index[str(order_id)] = order
            if client_order_id is not None:
                order_index[str(client_order_id)] = order

        for order in orders:
            _index_order(order)

        # Update persisted records that are still using submit_id as order_id, now that
        # the broker has returned the canonical order_id.
        if self._orders_store is not None and pending_by_client_id:
            now = datetime.now(timezone.utc)
            for order in orders:
                order_id = self._get_order_field(order, "order_id", "id")
                client_order_id = self._get_order_field(order, "client_order_id", "client_id")
                if order_id is None or client_order_id is None:
                    continue
                client_id_str = str(client_order_id)
                pending_record = pending_by_client_id.get(client_id_str)
                if pending_record is None:
                    continue
                if pending_record.order_id != pending_record.client_order_id:
                    continue
                order_id_str = str(order_id)
                if order_id_str == pending_record.order_id:
                    continue
                try:
                    updated = replace(
                        pending_record,
                        order_id=order_id_str,
                        status=PersistedOrderStatus.OPEN,
                        updated_at=now,
                        metadata={
                            **(pending_record.metadata or {}),
                            "source": "order_audit",
                            "note": "normalized_submit_id_to_order_id",
                        },
                    )
                    await self._broker_calls(self._orders_store.upsert_by_client_id, updated)
                except Exception as exc:
                    logger.warning(
                        "Failed to persist order_id normalization",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        operation="order_reconciliation",
                        stage="persist_normalization",
                        client_order_id=client_id_str,
                        order_id=order_id_str,
                    )

        if pending:
            open_ids = set(order_index.keys())
            missing_records = [
                record
                for record in pending
                if str(record.order_id) not in open_ids
                and str(record.client_order_id) not in open_ids
            ]
            refreshed_orders = await self._refresh_missing_persisted_orders(
                missing_records,
                bot_id=bot_id,
            )
            for refreshed_order in refreshed_orders:
                _index_order(refreshed_order)

        # Reconcile the tracked IDs: replace submit_ids with canonical order_ids and drop
        # IDs that are no longer open at the broker.
        snapshot = list(self._open_orders)
        snapshot_set = set(snapshot)
        reconciled: list[str] = []
        seen: set[str] = set()

        def _canonical_order_id(tracked_id: str) -> str | None:
            order = order_index.get(tracked_id)
            if order is None:
                return None
            oid = self._get_order_field(order, "order_id", "id") or tracked_id
            return str(oid)

        for tracked_id in snapshot:
            canonical = _canonical_order_id(tracked_id)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            reconciled.append(canonical)

        # Preserve any IDs appended while we were reconciling (e.g., order submission)
        # without losing them via slice assignment.
        for tracked_id in list(self._open_orders):
            if tracked_id in snapshot_set:
                continue
            canonical = _canonical_order_id(tracked_id) or str(tracked_id)
            if canonical in seen:
                continue
            seen.add(canonical)
            reconciled.append(canonical)

        if reconciled != snapshot:
            self._open_orders[:] = reconciled

        # Drift detection: bot-owned open orders present at the broker but not tracked in
        # the store/open-orders list can indicate "ghost" orders after a crash.
        unknown_bot_orders: list[str] = []
        unknown_orders: list[Any] = []
        for order in orders:
            client_order_id = self._get_order_field(order, "client_order_id", "client_id")
            if client_order_id is None:
                continue
            client_id_str = str(client_order_id)
            if not client_id_str.startswith(prefix):
                continue
            order_id = self._get_order_field(order, "order_id", "id")
            order_id_str = str(order_id) if order_id is not None else client_id_str
            if order_id_str in tracked_ids or client_id_str in tracked_ids:
                continue
            unknown_bot_orders.append(order_id_str)
            unknown_orders.append(order)

        recovered_records = await self._recover_unknown_bot_orders(
            unknown_orders,
            bot_id=bot_id,
        )
        if recovered_records:
            recovered_ids = {record.order_id for record in recovered_records}
            recovered_ids.update(record.client_order_id for record in recovered_records)
            tracked_ids.update(recovered_ids)
            unknown_bot_orders = [
                order_id for order_id in unknown_bot_orders if order_id not in recovered_ids
            ]

        if not unknown_bot_orders:
            self._order_reconciliation_drift_failures = 0
            self._order_reconciliation_drift_escalated = False
            return

        # Record drift event for observability.
        payload = {
            "timestamp": time.time(),
            "bot_id": bot_id,
            "unknown_bot_order_ids": unknown_bot_orders[:10],
            "unknown_bot_order_count": len(unknown_bot_orders),
            "open_order_count": len(orders),
        }
        self._append_event("order_reconciliation_drift", payload)
        logger.warning(
            "Order reconciliation drift detected",
            unknown_bot_order_count=len(unknown_bot_orders),
            open_order_count=len(orders),
            operation="order_reconciliation",
            stage="drift",
        )

        if self._order_reconciliation_drift_escalated:
            return

        self._order_reconciliation_drift_failures += 1
        if self._order_reconciliation_drift_failures < ORDER_RECONCILIATION_DRIFT_MAX_FAILURES:
            return

        self._order_reconciliation_drift_escalated = True
        await self._handle_order_reconciliation_drift(unknown_bot_orders)

    async def _handle_order_reconciliation_drift(self, order_ids: list[str]) -> None:
        """Escalate persistent order-reconciliation drift to graceful degradation."""
        risk_manager = self.context.risk_manager
        config = risk_manager.config if risk_manager else None

        cooldown_seconds = getattr(config, "api_health_cooldown_seconds", 300) if config else 300
        if risk_manager is not None:
            risk_manager.set_reduce_only_mode(True, reason="order_reconciliation_drift")

        broker = self.context.broker
        if broker is not None:
            cancel_executor = BrokerExecutor(broker=broker)
            retry_enabled = getattr(self.context.config, "order_submission_retries_enabled", False)
            for order_id in order_ids:
                try:
                    await self._broker_calls(
                        cancel_executor.cancel_order,
                        order_id,
                        use_retry=retry_enabled,
                        allow_idempotent=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to cancel drifted order",
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        operation="order_reconciliation",
                        stage="cancel",
                        order_id=order_id,
                    )

        self._degradation.pause_all(
            seconds=cooldown_seconds,
            reason="order_reconciliation_drift",
            allow_reduce_only=True,
        )

        # Record a synthetic rejection for telemetry.
        try:
            self._order_submitter.record_rejection(
                symbol="*",
                side="*",
                quantity=Decimal("0"),
                price=None,
                reason="order_reconciliation_drift",
            )
        except Exception:
            pass

        await self._notify(
            title="Order Reconciliation Drift - Trading Paused",
            message=(
                "Detected open broker orders that appear to belong to this bot but are not tracked. "
                f"Paused trading for {cooldown_seconds}s and enabled reduce-only mode. "
                f"Attempted cancellation for {len(order_ids)} order(s)."
            ),
            severity=AlertSeverity.ERROR,
            context={
                "unknown_order_ids": order_ids[:10],
                "unknown_order_count": len(order_ids),
                "cooldown_seconds": cooldown_seconds,
            },
        )

    # =========================================================================
    # Streaming Lifecycle Methods
    # =========================================================================

    def _should_enable_streaming(self) -> bool:
        """Check if streaming should be enabled based on config."""
        return _should_enable_streaming(self)

    def _schedule_coroutine(self, coro: Any) -> None:
        """Schedule a coroutine for execution."""
        _schedule_coroutine(self, coro)

    async def _start_streaming(self) -> asyncio.Task[Any] | None:
        """Start WebSocket streaming."""
        return await _start_streaming(self)

    async def _stop_streaming(self) -> None:
        """Stop WebSocket streaming."""
        await _stop_streaming(self)

    def _handle_stream_task_completion(self, task: asyncio.Task[Any]) -> None:
        """Handle stream task completion callback."""
        _handle_stream_task_completion(self, task)

    async def _run_stream_loop_async(
        self,
        symbols: list[str],
        level: int,
        stop_signal: threading.Event | None,
    ) -> None:
        """Run streaming loop asynchronously."""
        await _run_stream_loop_async(self, symbols, level, stop_signal)

    def _run_stream_loop(
        self,
        symbols: list[str],
        level: int,
        stop_signal: threading.Event | None,
    ) -> None:
        """Run streaming loop synchronously (called from executor)."""
        _run_stream_loop(self, symbols, level, stop_signal)

    def _extract_mark_from_message(self, msg: dict[str, Any]) -> Decimal | None:
        """Extract mark price from WebSocket message."""
        return extract_mark_from_message(msg)

    def _update_mark_and_metrics(
        self,
        ctx: CoordinatorContext,
        symbol: str,
        mark: Decimal,
    ) -> None:
        """Update mark price and related metrics."""
        update_mark_and_metrics(self, ctx, symbol, mark)

    @property
    def status_reporter(self) -> StatusReporter:
        return self._status_reporter

    @property
    def price_history(self) -> dict[str, deque[Decimal]]:
        """Access price history via PriceTickStore."""
        return self._price_tick_store.price_history

    async def _notify(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send notification if service is available."""
        if self.context.notification_service is None:
            return
        try:
            await self.context.notification_service.notify(
                title=title,
                message=message,
                severity=severity,
                source="TradingEngine",
                context=context,
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    @property
    def name(self) -> str:
        return "strategy"

    @property
    def state(self) -> EngineState:
        return self._lifecycle.state

    @property
    def running(self) -> bool:
        return self.state in (EngineState.STARTING, EngineState.RUNNING)

    @running.setter
    def running(self, value: bool) -> None:
        target = EngineState.RUNNING if value else EngineState.STOPPED
        self._lifecycle.transition(
            target,
            reason="running_override",
            details={"via": "running_set"},
            force=True,
        )

    def _transition_state(
        self,
        target: EngineState,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        return self._lifecycle.transition(
            target,
            reason=reason,
            details=details,
            force=force,
        )

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Start the main trading loop and heartbeat service.

        Before starting, attempts to rehydrate state from EventStore.
        """
        self._transition_state(EngineState.STARTING, reason="start_background_tasks")
        # Rehydrate state from EventStore before starting
        if not self._rehydrated:
            self._rehydrate_from_events()
            self._rehydrated = True

        self._record_runtime_start()

        self._transition_state(EngineState.RUNNING, reason="tasks_scheduled")

        tasks: list[asyncio.Task[Any]] = []

        # Start main trading loop
        trading_task = asyncio.create_task(self._run_loop())
        self._register_background_task(trading_task)
        tasks.append(trading_task)

        # Start heartbeat service
        heartbeat_task = await self._heartbeat.start()
        if heartbeat_task:
            self._register_background_task(heartbeat_task)
            tasks.append(heartbeat_task)

        # Start status reporter
        status_task = await self._status_reporter.start()
        if status_task:
            self._register_background_task(status_task)
            tasks.append(status_task)

        # Start health check runner for active /health probes
        await self._health_check_runner.start()

        # Start database pruning task via system maintenance service
        prune_task = await self._system_maintenance.start_prune_loop()
        self._register_background_task(prune_task)
        tasks.append(prune_task)

        # Start runtime guard sweep (daily loss, liquidation buffer, volatility)
        if self._guard_manager is not None:
            guard_task = asyncio.create_task(
                self._runtime_guard_sweep(), name="runtime_guard_sweep"
            )
            self._register_background_task(guard_task)
            tasks.append(guard_task)

        # Start WebSocket streaming if enabled
        if self._should_enable_streaming():
            start_streaming_background(self)
            logger.info(
                "Started WebSocket streaming",
                operation="streaming",
                stage="start",
            )

        # Start WS health watchdog
        self._ws_health_task = asyncio.create_task(
            self._monitor_ws_health(), name="ws_health_watchdog"
        )
        self._register_background_task(self._ws_health_task)
        tasks.append(self._ws_health_task)
        logger.info(
            "Started WS health watchdog",
            operation="ws_health",
            stage="start",
        )

        return tasks

    def _rehydrate_from_events(self) -> int:
        """Restore price history from persisted events.

        Delegates to PriceTickStore for the actual rehydration logic.

        Returns:
            Number of price ticks restored
        """
        # Prepare strategy rehydration callback if strategy supports it
        strategy_callback = None
        if hasattr(self.strategy, "rehydrate"):
            strategy_callback = self.strategy.rehydrate

        return self._price_tick_store.rehydrate(strategy_rehydrate_callback=strategy_callback)

    def _record_runtime_start(self) -> None:
        event_store = getattr(self, "_event_store", None)
        if event_store is None:
            return

        try:
            package_version = metadata.version("gpt-trader")
        except metadata.PackageNotFoundError:
            package_version = None

        payload = {
            "timestamp": time.time(),
            "profile": str(self.context.config.profile or ""),
            "bot_id": self.context.bot_id or None,
            "build_sha": os.getenv("GPT_TRADER_BUILD_SHA"),
            "package_version": package_version,
            "python_version": sys.version.split()[0],
            "pid": os.getpid(),
        }
        try:
            event_store.append("runtime_start", payload)
        except Exception:
            logger.exception("Failed to record runtime_start", operation="runtime_start")

    async def _runtime_guard_sweep(self) -> None:
        """Periodically run runtime guards to check risk limits.

        Runs on a cadence to proactively detect risk breaches (daily loss,
        liquidation buffer, volatility) rather than only at order time.

        On guard failure, triggers graceful degradation (pause + reduce-only).
        """
        interval = getattr(self.context.config, "runtime_guard_interval", 60)
        while self.running:
            try:
                if self._guard_manager is not None:
                    # Use run_runtime_guards directly to catch GuardError for degradation
                    state = await self._broker_calls(self._guard_manager.run_runtime_guards)
                    self._record_guard_events(state.guard_events)

            except GuardError as err:
                # Trigger graceful degradation on guard failure
                await self._handle_guard_failure(err)

            except Exception:
                logger.exception("Runtime guard sweep failed", operation="runtime_guards")

            await asyncio.sleep(interval)

    async def _handle_guard_failure(self, err: GuardError) -> None:
        """Handle guard failure by triggering graceful degradation."""
        risk_manager = self.context.risk_manager
        config = risk_manager.config if risk_manager else None
        self._record_guard_failure_event(err)

        # Determine cooldown from config
        cooldown_seconds = 300  # Default 5 minutes
        if config is not None:
            cooldown_seconds = config.api_health_cooldown_seconds

        # Set reduce-only mode
        if risk_manager is not None:
            risk_manager.set_reduce_only_mode(True, reason=f"guard_failure:{err.guard_name}")

        # Cancel all open orders
        if self._guard_manager is not None:
            cancelled = await self._broker_calls(self._guard_manager.cancel_all_orders)
            logger.warning(
                "Guard failure triggered order cancellation",
                guard_name=err.guard_name,
                cancelled_orders=cancelled,
                operation="degradation",
                stage="cancel_orders",
            )

        # Pause all trading
        self._degradation.pause_all(
            seconds=cooldown_seconds,
            reason=f"guard_failure:{err.guard_name}",
            allow_reduce_only=True,
        )

        # Record rejection for telemetry
        self._order_submitter.record_rejection(
            symbol="*",
            side="*",
            quantity=Decimal("0"),
            price=None,
            reason=f"guard_failure:{err.guard_name}",
        )

        # Notify
        await self._notify(
            title="Guard Failure - Trading Paused",
            message=f"Runtime guard '{err.guard_name}' failed: {err.message}. "
            f"Trading paused for {cooldown_seconds}s. Reduce-only mode activated.",
            severity=AlertSeverity.ERROR,
            context={
                "guard_name": err.guard_name,
                "message": err.message,
                "cooldown_seconds": cooldown_seconds,
                "recoverable": err.recoverable,
            },
        )

    async def _monitor_ws_health(self) -> None:
        """Monitor WebSocket health and trigger degradation on staleness.

        Periodically polls WS health metrics from the broker. If messages
        or heartbeats are stale beyond configured thresholds, triggers:
        - Reduce-only mode for affected symbols
        - Symbol pause for configured cooldown
        - Notification alerts

        On reconnect, pauses briefly to allow state synchronization.
        """
        risk_manager = self.context.risk_manager
        config = getattr(risk_manager, "config", None) if risk_manager else None

        def _coerce_seconds(value: Any, default: float) -> float:
            if value is None or isinstance(value, bool):
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return default

        # Get thresholds from config or use defaults
        interval = _coerce_seconds(getattr(config, "ws_health_interval_seconds", None), 5.0)
        message_stale_threshold = _coerce_seconds(
            getattr(config, "ws_message_stale_seconds", None), 15.0
        )
        heartbeat_stale_threshold = _coerce_seconds(
            getattr(config, "ws_heartbeat_stale_seconds", None), 30.0
        )
        reconnect_pause = _coerce_seconds(getattr(config, "ws_reconnect_pause_seconds", None), 30.0)

        interval = max(0.1, interval)
        message_stale_threshold = max(0.0, message_stale_threshold)
        heartbeat_stale_threshold = max(0.0, heartbeat_stale_threshold)
        reconnect_pause = max(0.0, reconnect_pause)

        last_reconnect_count = 0

        while self.running:
            try:
                # Get WS health from broker (if it supports the method)
                broker = self.context.broker
                ws_health: dict[str, Any] = {}

                if broker is not None and hasattr(broker, "get_ws_health"):
                    try:
                        ws_health = broker.get_ws_health()
                    except Exception as exc:
                        logger.debug(
                            "Failed to get WS health",
                            error=str(exc),
                            operation="ws_health",
                            stage="poll",
                        )

                if not ws_health:
                    # No WS connection or broker doesn't support health check
                    await asyncio.sleep(interval)
                    continue

                current_time = time.time()

                last_message_ts_raw = ws_health.get("last_message_ts")
                last_message_ts = (
                    float(last_message_ts_raw)
                    if isinstance(last_message_ts_raw, (int, float))
                    and not isinstance(last_message_ts_raw, bool)
                    else None
                )
                last_heartbeat_ts_raw = ws_health.get("last_heartbeat_ts")
                last_heartbeat_ts = (
                    float(last_heartbeat_ts_raw)
                    if isinstance(last_heartbeat_ts_raw, (int, float))
                    and not isinstance(last_heartbeat_ts_raw, bool)
                    else None
                )

                reconnect_count_raw = ws_health.get("reconnect_count", 0)
                reconnect_count = (
                    int(reconnect_count_raw)
                    if isinstance(reconnect_count_raw, (int, float))
                    and not isinstance(reconnect_count_raw, bool)
                    else 0
                )
                gap_count_raw = ws_health.get("gap_count", 0)
                gap_count = (
                    int(gap_count_raw)
                    if isinstance(gap_count_raw, (int, float))
                    and not isinstance(gap_count_raw, bool)
                    else 0
                )

                connected_raw = ws_health.get("connected", False)
                connected = connected_raw if isinstance(connected_raw, bool) else False

                # Check for reconnect event
                if reconnect_count > last_reconnect_count:
                    logger.warning(
                        "WebSocket reconnected - pausing for state sync",
                        reconnect_count=reconnect_count,
                        pause_seconds=reconnect_pause,
                        operation="ws_health",
                        stage="reconnect",
                    )
                    self._append_event(
                        "websocket_reconnect",
                        {
                            "reconnect_count": reconnect_count,
                            "gap_count": gap_count,
                            "connected": connected,
                            "timestamp": current_time,
                        },
                    )
                    last_reconnect_count = reconnect_count

                    # Reset reconnect attempts on successful reconnect
                    self._ws_reconnect_attempts = 0
                    self._ws_reconnect_delay = 1.0

                    if self._user_event_handler is not None:
                        backfill = getattr(self._user_event_handler, "request_backfill", None)
                        if callable(backfill):
                            backfill(reason="ws_reconnect", run_in_thread=True)

                    # Pause all symbols briefly after reconnect
                    self._degradation.pause_all(
                        seconds=reconnect_pause,
                        reason="ws_reconnect",
                        allow_reduce_only=True,
                    )

                    await self._notify(
                        title="WebSocket Reconnected",
                        message=f"Trading paused for {reconnect_pause}s for state sync.",
                        severity=AlertSeverity.WARNING,
                        context={"reconnect_count": reconnect_count},
                    )

                    await asyncio.sleep(interval)
                    continue

                # Check message staleness
                is_message_stale = False
                if last_message_ts is not None:
                    message_age = current_time - last_message_ts
                    is_message_stale = message_age > message_stale_threshold

                # Check heartbeat staleness
                is_heartbeat_stale = False
                if last_heartbeat_ts is not None:
                    heartbeat_age = current_time - last_heartbeat_ts
                    is_heartbeat_stale = heartbeat_age > heartbeat_stale_threshold

                # Trigger degradation if stale
                if is_message_stale or is_heartbeat_stale:
                    stale_reason = "ws_message_stale" if is_message_stale else "ws_heartbeat_stale"
                    stale_age = (
                        (current_time - last_message_ts)
                        if is_message_stale and last_message_ts
                        else (current_time - last_heartbeat_ts if last_heartbeat_ts else 0)
                    )

                    logger.warning(
                        "WebSocket data stale - triggering degradation",
                        reason=stale_reason,
                        stale_age_seconds=stale_age,
                        message_stale=is_message_stale,
                        heartbeat_stale=is_heartbeat_stale,
                        connected=connected,
                        gap_count=gap_count,
                        operation="ws_health",
                        stage="degradation",
                    )

                    # Set reduce-only mode
                    if risk_manager is not None:
                        risk_manager.set_reduce_only_mode(True, reason=stale_reason)

                    # Pause all trading (allow reduce-only)
                    cooldown = reconnect_pause
                    self._degradation.pause_all(
                        seconds=cooldown,
                        reason=stale_reason,
                        allow_reduce_only=True,
                    )

                    await self._notify(
                        title="WebSocket Stale - Trading Paused",
                        message=f"No WS data for {stale_age:.1f}s. Reduce-only mode enabled.",
                        severity=AlertSeverity.WARNING,
                        context={
                            "reason": stale_reason,
                            "stale_age_seconds": stale_age,
                            "cooldown_seconds": cooldown,
                        },
                    )

                # Log gap detection warnings
                if gap_count > 0 and self._cycle_count % 60 == 0:
                    logger.info(
                        "WebSocket sequence gaps detected",
                        gap_count=gap_count,
                        operation="ws_health",
                        stage="info",
                    )

                # Update status reporter with WS health
                self._status_reporter.update_ws_health(ws_health)

            except Exception:
                logger.exception("WS health watchdog error", operation="ws_health")

            await asyncio.sleep(interval)

    async def _run_loop(self) -> None:
        logger.info("Starting strategy loop...")
        while self.running:
            try:
                await self._cycle()
                # Record successful cycle
                self._status_reporter.record_cycle()
            except Exception as e:
                logger.error(f"Error in strategy cycle: {e}", exc_info=True)
                # Record error in status reporter
                self._status_reporter.record_error(str(e))
                await self._notify(
                    title="Strategy Cycle Error",
                    message=f"Error during trading cycle: {e}",
                    severity=AlertSeverity.ERROR,
                    context={"error": str(e)},
                )

            await asyncio.sleep(self.context.config.interval)

    def _report_system_status(self) -> None:
        """Collect and report system health metrics.

        Delegates to SystemMaintenanceService for the actual reporting.
        """
        self._system_maintenance.report_system_status(
            latency_seconds=self._last_latency,
            connection_status=self._connection_status,
        )

    async def _cycle(self) -> None:
        """One trading cycle."""
        assert self.context.broker is not None, "Broker not initialized"
        self._cycle_count += 1

        # Wrap entire cycle in correlation context and trace span
        start_time = time.perf_counter()
        result = "ok"
        with correlation_context(cycle=self._cycle_count):
            with trace_span("cycle", {"cycle": self._cycle_count}) as span:
                try:
                    await self._cycle_inner()
                except Exception:
                    result = "error"
                    if span:
                        span.set_attribute("error", True)
                    raise
                finally:
                    duration = time.perf_counter() - start_time
                    if span:
                        span.set_attribute("duration_seconds", duration)
                        span.set_attribute("result", result)
                    record_histogram(
                        "gpt_trader_cycle_duration_seconds",
                        duration,
                        labels={"result": result},
                    )

    async def _cycle_inner(self) -> None:
        """Inner cycle logic wrapped in correlation context."""
        logger.info(f"=== CYCLE {self._cycle_count} START ===")

        # Report system status at start of cycle
        self._report_system_status()
        broker = self.context.broker
        if broker is None:
            logger.error("Broker not initialized", operation="cycle")
            self._connection_status = "DISCONNECTED"
            return

        positions, audit_task = await self._fetch_positions_and_audit()
        equity = await self._compute_equity(positions)
        if equity is None:
            await self._await_audit_task(audit_task, context="during equity error path")
            return

        await self._await_audit_task(audit_task, context="post equity")
        self._update_equity_and_risk(equity)

        # Ensure symbols is a list to avoid iterator exhaustion during multiple iterations
        symbols = list(self.context.config.symbols)
        tickers = await self._fetch_batch_tickers(broker, symbols)

        tasks = [
            self._process_symbol(
                symbol=symbol,
                broker=broker,
                ticker=tickers.get(symbol),
                positions=positions,
                equity=equity,
            )
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures: list[Exception] = []

        for symbol, res in zip(symbols, results):
            if isinstance(res, Exception):
                logger.error(
                    f"Failed to process symbol {symbol}: {res}",
                    exc_info=res,
                    symbol=symbol,
                )
                failures.append(res)

        if failures:
            raise ExceptionGroup("Cycle completed with symbol processing errors", failures)

    async def _fetch_positions_and_audit(
        self,
    ) -> tuple[dict[str, Position], asyncio.Task[None]]:
        logger.info("Step 1: Fetching positions and auditing orders (parallel)...")
        positions_task = asyncio.create_task(self._fetch_positions())
        if getattr(self.context.config, "dry_run", False):
            logger.info(
                "Dry-run enabled; skipping order audit",
                operation="order_audit",
                stage="skip",
            )
            audit_task = asyncio.create_task(asyncio.sleep(0))
        else:
            audit_task = asyncio.create_task(self._audit_orders())

        with profile_span("fetch_positions") as _pos_span:
            positions = await positions_task
        self._current_positions = positions
        logger.info(f"Fetched {len(positions)} positions")

        self._status_reporter.update_positions(self._positions_to_status_format(positions))
        return positions, audit_task

    async def _await_audit_task(self, task: asyncio.Task, *, context: str) -> None:
        try:
            await task
        except Exception as e:
            logger.warning(f"Order audit failed {context}: {e}")

    async def _compute_equity(self, positions: dict[str, Position]) -> Decimal | None:
        logger.info("Step 2: Calculating total equity...")
        with profile_span("equity_computation") as _eq_span:
            equity = await self._fetch_total_equity(positions)
        if equity is None:
            logger.error(
                "Failed to fetch equity - cannot continue cycle. "
                "Check logs above for balance fetch errors."
            )
            self._status_reporter.record_error("Failed to fetch equity")
            return None
        return equity

    def _update_equity_and_risk(self, equity: Decimal) -> None:
        logger.info(f"Successfully calculated equity: ${equity}")
        self._status_reporter.update_equity(equity)
        logger.info("Equity updated in status reporter")

        if self.context.risk_manager:
            triggered = self.context.risk_manager.track_daily_pnl(equity, {})
            if triggered:
                logger.warning("Daily loss limit triggered! Reduce-only mode activated.")

            rm = self.context.risk_manager
            daily_loss_pct = 0.0
            start_equity = getattr(rm, "_start_of_day_equity", 0)
            if start_equity and start_equity > 0:
                daily_pnl = equity - start_equity
                daily_loss_pct = float(-daily_pnl / start_equity)

            self._status_reporter.update_risk(
                max_leverage=float(getattr(rm.config, "max_leverage", 0.0) if rm.config else 0.0),
                daily_loss_limit=float(
                    getattr(rm.config, "daily_loss_limit_pct", 0.0) if rm.config else 0.0
                ),
                current_daily_loss=daily_loss_pct,
                reduce_only=getattr(rm, "_reduce_only_mode", False),
                reduce_reason=getattr(rm, "_reduce_only_reason", ""),
            )

    async def _fetch_batch_tickers(
        self, broker: Any, symbols: list[str]
    ) -> dict[str, dict[str, Any]]:
        tickers: dict[str, dict[str, Any]] = {}
        batch_start = time.time()

        get_tickers_method = getattr(broker, "get_tickers", None)
        if get_tickers_method is not None and callable(get_tickers_method):
            try:
                result = await self._broker_calls(get_tickers_method, symbols)
                if isinstance(result, dict):
                    tickers = result
                    logger.debug(
                        f"Batch ticker fetch: {len(tickers)}/{len(symbols)} symbols "
                        f"in {time.time() - batch_start:.3f}s"
                    )
            except Exception as e:
                logger.warning(f"Batch ticker fetch failed, falling back to individual: {e}")

        return tickers

    async def _process_symbol(
        self,
        *,
        symbol: str,
        broker: Any,
        ticker: dict[str, Any] | None,
        positions: dict[str, Position],
        equity: Decimal,
    ) -> None:
        candles: list[Any] = []
        start_time = time.time()

        if ticker is None:
            try:
                ticker = await self._broker_calls(broker.get_ticker, symbol)
            except Exception as e:
                logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                self._connection_status = "DISCONNECTED"
                return

        if ticker is None or not ticker.get("price"):
            logger.error(f"No ticker data for {symbol}")
            self._connection_status = "DISCONNECTED"
            return

        try:
            candles_result = await self._broker_calls(
                broker.get_candles,
                symbol,
                granularity="ONE_MINUTE",
            )
            if isinstance(candles_result, Exception):
                logger.warning(f"Failed to fetch candles for {symbol}: {candles_result}")
            else:
                candles = candles_result or []
        except Exception as e:
            logger.warning(f"Failed to fetch candles for {symbol}: {e}")

        self._last_latency = time.time() - start_time
        self._connection_status = "CONNECTED"

        price = Decimal(str(ticker.get("price", 0)))
        logger.info(f"{symbol} price: {price}")

        if self.context.risk_manager is not None:
            self.context.risk_manager.last_mark_update[symbol] = time.time()

        self._status_reporter.update_price(symbol, price)
        self._record_price_tick(symbol, price)

        position_state = self._build_position_state(symbol, positions)
        with profile_span("strategy_decision", {"symbol": symbol}) as _strat_span:
            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=price,
                position_state=position_state,
                recent_marks=self.price_history[symbol],
                equity=equity,
                product=None,
                candles=candles,
            )

        logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

        active_strats = getattr(
            self.strategy, "active_strategies", [self.strategy.__class__.__name__]
        )
        decision_record = {
            "symbol": symbol,
            "action": decision.action.value,
            "reason": decision.reason,
            "confidence": str(decision.confidence),
            "timestamp": time.time(),
        }
        self._status_reporter.update_strategy(active_strats, [decision_record])

        await self._handle_decision(
            symbol=symbol,
            decision=decision,
            price=price,
            equity=equity,
            position_state=position_state,
        )

    async def _handle_decision(
        self,
        *,
        symbol: str,
        decision: Decision,
        price: Decimal,
        equity: Decimal,
        position_state: dict[str, Any] | None,
    ) -> None:
        if decision.action in (Action.BUY, Action.SELL):
            logger.info(
                "Executing order",
                symbol=symbol,
                action=decision.action.value,
                operation="order_placement",
                stage="start",
            )
            try:
                with profile_span(
                    "order_placement", {"symbol": symbol, "action": decision.action.value}
                ):
                    result = await self._validate_and_place_order(
                        symbol=symbol,
                        decision=decision,
                        price=price,
                        equity=equity,
                    )
                if result.blocked:
                    logger.warning(
                        "Order blocked",
                        symbol=symbol,
                        action=decision.action.value,
                        reason=result.reason,
                        operation="order_placement",
                        stage="blocked",
                    )
                elif result.failed:
                    logger.error(
                        "Order submission failed",
                        symbol=symbol,
                        action=decision.action.value,
                        reason=result.reason,
                        error_message=result.error,
                        operation="order_placement",
                        stage="failed",
                    )
                    failure_detail = result.error or result.reason or "unknown"
                    await self._notify(
                        title="Order Submission Failed",
                        message=(
                            f"Failed to submit {decision.action.value} order for {symbol}: "
                            f"{failure_detail}"
                        ),
                        severity=AlertSeverity.ERROR,
                        context={
                            "symbol": symbol,
                            "action": decision.action.value,
                            "reason": result.reason,
                            "error": result.error,
                        },
                    )
            except Exception as e:
                logger.error(
                    "Order placement failed",
                    symbol=symbol,
                    action=decision.action.value,
                    error_message=str(e),
                    operation="order_placement",
                    stage="failed",
                )
                await self._notify(
                    title="Order Placement Failed",
                    message=f"Failed to execute {decision.action} for {symbol}: {e}",
                    severity=AlertSeverity.ERROR,
                    context={
                        "symbol": symbol,
                        "action": decision.action.value,
                        "error": str(e),
                    },
                )
        elif decision.action == Action.CLOSE and position_state:
            logger.info(f"CLOSE signal for {symbol} - not fully implemented yet")

    async def _fetch_total_equity(self, positions: dict[str, Position]) -> Decimal | None:
        """Fetch total equity = collateral + unrealized PnL."""
        return await self._equity_calculator.calculate_total_equity(self.context.broker, positions)

    async def _fetch_positions(self) -> dict[str, Position]:
        """Fetch current positions as a lookup dict."""
        assert self.context.broker is not None
        start_time = time.perf_counter()
        result = "ok"
        try:
            positions_list = await self._broker_calls(self.context.broker.list_positions)
            # Success: reset broker failure counter
            self._degradation.reset_broker_failures()
            return {p.symbol: p for p in positions_list}
        except Exception as e:
            result = "error"
            logger.error(f"Failed to fetch positions: {e}")
            # Track broker failure for degradation
            config = self.context.risk_manager.config if self.context.risk_manager else None
            if config is not None:
                self._degradation.record_broker_failure(config)
            return {}
        finally:
            duration = time.perf_counter() - start_time
            record_histogram(
                "gpt_trader_positions_fetch_seconds",
                duration,
                labels={"result": result},
            )

    def _build_position_state(
        self, symbol: str, positions: dict[str, Position]
    ) -> dict[str, Any] | None:
        """Build position state dict for strategy.decide()."""
        if symbol not in positions:
            return None
        pos = positions[symbol]
        return {
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "side": pos.side,
            # Add other fields if needed by strategy
        }

    def _record_price_tick(self, symbol: str, price: Decimal) -> None:
        """Persist price tick to EventStore for crash recovery.

        Delegates to PriceTickStore which handles both in-memory
        history update and EventStore persistence.
        """
        self._price_tick_store.record_price_tick(symbol, price)

    def _positions_to_risk_format(
        self, positions: dict[str, Position]
    ) -> dict[str, dict[str, Any]]:
        """Convert Position objects to dict format expected by risk manager."""
        return {
            symbol: {
                "quantity": pos.quantity,
                "mark": pos.mark_price,
            }
            for symbol, pos in positions.items()
        }

    def _positions_to_status_format(
        self, positions: dict[str, Position]
    ) -> dict[str, dict[str, Any]]:
        """Convert Position objects to dict format for StatusReporter with complete TUI data."""
        return {
            symbol: {
                "quantity": str(pos.quantity),
                "mark_price": str(pos.mark_price),
                "entry_price": str(pos.entry_price),
                "unrealized_pnl": str(pos.unrealized_pnl),
                "realized_pnl": str(pos.realized_pnl),
                "side": pos.side,
            }
            for symbol, pos in positions.items()
        }

    def _calculate_order_quantity(
        self,
        symbol: str,
        price: Decimal,
        equity: Decimal,
        product: Product | None,
        *,
        quantity_override: Decimal | None = None,
    ) -> Decimal:
        """Calculate order size based on equity and position_fraction."""
        # External override (submit_order) bypasses dynamic sizing.
        if quantity_override is not None:
            return quantity_override

        # 1. Determine fraction
        fraction = Decimal("0.1")  # Default
        if hasattr(self.strategy, "config") and self.strategy.config.position_fraction:
            fraction = Decimal(str(self.strategy.config.position_fraction))
        elif (
            hasattr(self.context.config, "perps_position_fraction")
            and self.context.config.perps_position_fraction is not None
        ):
            fraction = Decimal(str(self.context.config.perps_position_fraction))

        # 2. Calculate raw quantity
        if price == 0:
            return Decimal("0")

        target_notional = equity * fraction
        quantity = target_notional / price

        # 3. Apply constraints
        if product and product.min_size:
            if quantity < product.min_size:
                logger.warning(f"Quantity {quantity} below min size {product.min_size}")
                return Decimal("0")

            # Round to step size if needed (simplified)
            # quantity = (quantity // product.step_size) * product.step_size

        return quantity

    def _is_reduce_only_order(self, current_pos: Position | None, side: OrderSide) -> bool:
        """Determine if an order would reduce an existing position."""
        if current_pos is None:
            return False

        # Handle Position objects
        if hasattr(current_pos, "side") and hasattr(current_pos, "quantity"):
            pos_side = current_pos.side.lower() if current_pos.side else ""
            pos_qty = current_pos.quantity
            # Reducing = LONG + SELL or SHORT + BUY
            return (pos_side == "long" and side == OrderSide.SELL and pos_qty > 0) or (
                pos_side == "short" and side == OrderSide.BUY and pos_qty > 0
            )

        # Handle dict format
        if isinstance(current_pos, dict):
            pos_side = str(current_pos.get("side", "")).lower()
            pos_qty = Decimal(str(current_pos.get("quantity", 0)))
            if pos_side in ("long", "short"):
                return (pos_side == "long" and side == OrderSide.SELL and pos_qty > 0) or (
                    pos_side == "short" and side == OrderSide.BUY and pos_qty > 0
                )
            # Legacy: quantity sign indicates direction
            return (pos_qty > 0 and side == OrderSide.SELL) or (
                pos_qty < 0 and side == OrderSide.BUY
            )

        return False

    def _record_decision_trace(self, trace: OrderDecisionTrace) -> None:
        """Persist the decision trace for auditability."""
        try:
            self._order_submitter.record_decision_trace(trace)
        except Exception as exc:
            logger.error(
                "Failed to record order decision trace",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="order_decision_trace",
                symbol=trace.symbol,
                side=trace.side,
            )

    def _append_event(self, event_type: str, data: dict[str, Any]) -> None:
        event_store = getattr(self, "_event_store", None)
        if event_store is None:
            return
        payload = dict(data)
        if "bot_id" not in payload and self.context.bot_id:
            payload["bot_id"] = self.context.bot_id
        try:
            event_store.append(event_type, payload)
        except Exception as exc:
            logger.error(
                "Failed to append event",
                event_type=event_type,
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="event_store",
            )

    def _record_guard_events(self, guard_events: list[dict[str, Any]]) -> None:
        if not guard_events:
            return
        for event in guard_events:
            if not event:
                continue
            payload = dict(event)
            if payload.get("triggered") is False:
                continue
            payload.setdefault("guard", "volatility_circuit_breaker")
            payload.setdefault("timestamp", time.time())
            self._append_event("guard_triggered", payload)

    def _record_guard_failure_event(self, err: GuardError) -> None:
        payload = {
            "guard": err.guard_name,
            "reason": err.message,
            "recoverable": err.recoverable,
            "category": getattr(err, "category", "unknown"),
            "details": err.details,
            "timestamp": time.time(),
            "triggered": True,
        }
        self._append_event("guard_triggered", payload)
        if err.guard_name == "api_health":
            api_payload = {
                "reason": err.message,
                "details": err.details,
                "timestamp": payload["timestamp"],
            }
            self._append_event("api_error", api_payload)

    def _get_order_field(self, order: Any, *keys: str) -> Any:
        if isinstance(order, dict):
            for key in keys:
                if key in order and order[key] is not None:
                    return order[key]
            return None
        for key in keys:
            if hasattr(order, key):
                value = getattr(order, key)
                if value is not None:
                    return value
        return None

    def _normalize_persisted_status(self, status: Any) -> PersistedOrderStatus:
        value = status.value if hasattr(status, "value") else status
        normalized = str(value).lower()
        mapping = {
            "pending": PersistedOrderStatus.PENDING,
            "submitted": PersistedOrderStatus.OPEN,
            "open": PersistedOrderStatus.OPEN,
            "partially_filled": PersistedOrderStatus.PARTIALLY_FILLED,
            "filled": PersistedOrderStatus.FILLED,
            "cancelled": PersistedOrderStatus.CANCELLED,
            "canceled": PersistedOrderStatus.CANCELLED,
            "rejected": PersistedOrderStatus.REJECTED,
            "expired": PersistedOrderStatus.EXPIRED,
            "failed": PersistedOrderStatus.FAILED,
        }
        return mapping.get(normalized, PersistedOrderStatus.OPEN)

    def _parse_decimal(self, value: Any, default: Decimal) -> Decimal:
        if value is None:
            return default
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return default

    def _parse_decimal_optional(self, value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return None

    def _merge_metadata(
        self,
        base: dict[str, Any] | None,
        update: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if base is None and update is None:
            return None
        merged = dict(base or {})
        if update:
            merged.update(update)
        return merged

    def _build_record_from_broker_order(
        self,
        order: Any,
        *,
        bot_id: str,
        now: datetime,
    ) -> OrderRecord | None:
        order_id = self._get_order_field(order, "order_id", "id")
        client_order_id = self._get_order_field(order, "client_order_id", "client_id") or order_id
        if order_id is None or client_order_id is None:
            return None
        symbol = self._get_order_field(order, "product_id", "symbol") or ""
        side_value = self._get_order_field(order, "side")
        side = str(side_value).lower() if side_value is not None else "unknown"
        order_type_value = self._get_order_field(order, "order_type", "type")
        order_type = str(order_type_value).lower() if order_type_value is not None else "unknown"
        quantity_value = self._get_order_field(order, "size", "quantity", "base_size")
        quantity = self._parse_decimal(quantity_value, Decimal("0"))
        price_value = self._get_order_field(order, "price")
        price = self._parse_decimal_optional(price_value)
        filled_value = self._get_order_field(order, "filled_size", "filled_quantity")
        filled_quantity = self._parse_decimal(filled_value, Decimal("0"))
        average_value = self._get_order_field(order, "average_filled_price", "avg_fill_price")
        average_fill_price = self._parse_decimal_optional(average_value)
        status_value = self._get_order_field(order, "status")
        status = self._normalize_persisted_status(status_value)
        created_value = self._get_order_field(
            order, "created_time", "created_at", "submitted_at", "created"
        )
        created_ts = self._parse_timestamp(created_value)
        created_at = datetime.fromtimestamp(created_ts, tz=timezone.utc)
        tif_value = self._get_order_field(order, "tif", "time_in_force")
        time_in_force = str(tif_value) if tif_value is not None else "GTC"
        metadata = {
            "source": "order_reconciliation",
            "raw_status": str(status_value or ""),
        }
        return OrderRecord(
            order_id=str(order_id),
            client_order_id=str(client_order_id),
            symbol=str(symbol),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status=status,
            filled_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            created_at=created_at,
            updated_at=now,
            bot_id=bot_id,
            time_in_force=time_in_force,
            metadata=metadata,
        )

    def _parse_timestamp(self, value: Any) -> float:
        if value is None:
            return time.time()
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, str):
            try:
                clean = value.rstrip("Z")
                dt = datetime.fromisoformat(clean)
                return dt.timestamp()
            except (ValueError, TypeError):
                return time.time()
        return time.time()

    def _record_unfilled_order_alerts(self, orders: list[Any]) -> None:
        risk_manager = self.context.risk_manager
        config = risk_manager.config if risk_manager else None
        threshold = getattr(config, "unfilled_order_alert_seconds", 300)
        if not orders or threshold <= 0:
            self._unfilled_order_alerts.clear()
            return

        now = time.time()
        active_ids: set[str] = set()

        for order in orders:
            order_id = self._get_order_field(
                order, "order_id", "id", "client_order_id", "client_id"
            )
            if order_id is None:
                continue
            order_id_str = str(order_id)
            active_ids.add(order_id_str)
            if order_id_str in self._unfilled_order_alerts:
                continue

            status = self._get_order_field(order, "status")
            status_str = (status.value if hasattr(status, "value") else str(status or "")).upper()
            if status_str in {"FILLED", "CANCELLED", "CANCELED", "REJECTED"}:
                continue

            created_at = self._get_order_field(
                order, "created_time", "created_at", "submitted_at", "created"
            )
            created_ts = self._parse_timestamp(created_at)
            age_seconds = now - created_ts
            if age_seconds < threshold:
                continue

            payload = {
                "order_id": order_id_str,
                "symbol": self._get_order_field(order, "product_id", "symbol") or "",
                "side": self._get_order_field(order, "side") or "",
                "status": status_str,
                "created_time": created_ts,
                "age_seconds": age_seconds,
                "threshold_seconds": threshold,
                "timestamp": now,
            }
            self._append_event("unfilled_order_alert", payload)
            self._unfilled_order_alerts[order_id_str] = now

        stale_ids = [
            order_id for order_id in self._unfilled_order_alerts if order_id not in active_ids
        ]
        for order_id in stale_ids:
            del self._unfilled_order_alerts[order_id]

    def _finalize_decision_trace(
        self,
        trace: OrderDecisionTrace,
        *,
        status: OrderSubmissionStatus,
        order_id: str | None = None,
        reason: str | None = None,
        error: str | None = None,
    ) -> OrderSubmissionResult:
        detail = reason or error
        trace.record_outcome("result", status.value, detail=detail)
        self._record_decision_trace(trace)
        return OrderSubmissionResult(
            status=status,
            order_id=order_id,
            reason=reason,
            error=error,
            decision_trace=trace,
        )

    async def _check_degradation_gate(
        self,
        *,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        trace: OrderDecisionTrace,
        reduce_only_flag: bool,
    ) -> OrderSubmissionResult | None:
        if self._degradation.is_paused(symbol, is_reduce_only=reduce_only_flag):
            pause_reason = self._degradation.get_pause_reason(symbol) or "unknown"
            logger.warning(
                f"Order blocked: trading paused for {symbol}",
                symbol=symbol,
                side=side.value,
                reason=pause_reason,
                operation="degradation",
                stage="order_blocked",
            )
            self._order_submitter.record_rejection(
                symbol, side.value, Decimal("0"), price, f"paused:{pause_reason}"
            )
            await self._notify(
                title="Order Blocked - Trading Paused",
                message=f"Cannot place {side.value} order for {symbol}: {pause_reason}",
                severity=AlertSeverity.WARNING,
                context={"symbol": symbol, "side": side.value, "reason": pause_reason},
            )
            trace.record_outcome("degradation_gate", "blocked", detail=pause_reason)
            return self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason=f"paused:{pause_reason}",
            )
        trace.record_outcome("degradation_gate", "passed")
        return None

    def _calculate_quantity_and_record(
        self,
        *,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        equity: Decimal,
        quantity_override: Decimal | None,
        trace: OrderDecisionTrace,
    ) -> tuple[Decimal, OrderSubmissionResult | None]:
        quantity = self._calculate_order_quantity(
            symbol,
            price,
            equity,
            product=None,
            quantity_override=quantity_override,
        )
        trace.quantity = quantity

        if quantity <= 0:
            logger.warning(f"Calculated quantity is {quantity}, skipping order")
            trace.record_outcome("sizing", "blocked", detail="quantity_zero")
            self._order_submitter.record_rejection(
                symbol, side.value, quantity, price, "quantity_zero"
            )
            return quantity, self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason="quantity_zero",
            )
        trace.record_outcome("sizing", "passed")
        return quantity, None

    async def _check_reduce_only_request(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        reduce_only_requested: bool,
        is_reducing: bool,
        trace: OrderDecisionTrace,
    ) -> OrderSubmissionResult | None:
        if reduce_only_requested and not is_reducing:
            logger.warning(
                "Reduce-only requested without a matching position",
                symbol=symbol,
                side=side.value,
                operation="reduce_only",
                stage="requested_not_reducing",
            )
            trace.record_outcome("reduce_only", "blocked", detail="requested_not_reducing")
            self._order_submitter.record_rejection(
                symbol, side.value, quantity, price, "reduce_only_not_reducing"
            )
            return self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason="reduce_only_not_reducing",
            )
        return None

    async def _run_security_validation(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        equity: Decimal,
        trace: OrderDecisionTrace,
    ) -> OrderSubmissionResult | None:
        from gpt_trader.security.validate import get_validator

        security_order = {
            "symbol": symbol,
            "side": side.value,
            "quantity": float(quantity),
            "price": float(price),
            "type": "MARKET",
        }

        limits = {}
        if hasattr(self.context.config, "risk"):
            risk = self.context.config.risk
            if risk:
                max_position_size = 0.05
                raw_max_position_fraction = getattr(risk, "max_position_pct", None)
                if raw_max_position_fraction is None:
                    raw_max_position_fraction = getattr(risk, "position_fraction", None)
                if raw_max_position_fraction is not None:
                    try:
                        max_position_size = float(raw_max_position_fraction)
                    except (TypeError, ValueError):
                        max_position_size = 0.05
                limits["max_position_size"] = max_position_size

                limits["max_leverage"] = float(getattr(risk, "max_leverage", 2.0))
                limits["max_daily_loss"] = float(getattr(risk, "daily_loss_limit_pct", 0.02))

        security_result = get_validator().validate_order_request(
            security_order, account_value=float(equity), limits=limits
        )

        if not security_result.is_valid:
            error_msg = f"Security validation failed: {', '.join(security_result.errors)}"
            logger.error(error_msg)
            self._order_submitter.record_rejection(
                symbol,
                side.value,
                quantity,
                price,
                "security_validation_failed",
            )
            await self._notify(
                title="Security Validation Failed",
                message=error_msg,
                severity=AlertSeverity.ERROR,
                context=security_order,
            )
            trace.record_outcome("security_validation", "blocked", detail=error_msg)
            return self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason=error_msg,
            )
        trace.record_outcome("security_validation", "passed")
        return None

    async def _apply_reduce_only_mode(
        self,
        *,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        quantity: Decimal,
        reduce_only_flag: bool,
        is_reducing: bool,
        current_pos: Position | dict[str, Any] | None,
        trace: OrderDecisionTrace,
    ) -> tuple[Decimal, OrderSubmissionResult | None]:
        risk_manager = self.context.risk_manager
        if risk_manager is None:
            logger.warning("No risk manager configured - skipping reduce-only checks")
            trace.record_outcome("reduce_only", "skipped")
            return quantity, None

        daily_pnl_triggered = bool(getattr(risk_manager, "_daily_pnl_triggered", False))
        reduce_only_mode = risk_manager.is_reduce_only_mode()
        reduce_only_active = reduce_only_mode or daily_pnl_triggered
        reduce_only_clamped = False
        if reduce_only_active and is_reducing and current_pos is not None:
            if hasattr(current_pos, "quantity"):
                current_qty = abs(current_pos.quantity)
            elif isinstance(current_pos, dict):
                current_qty = abs(Decimal(str(current_pos.get("quantity", 0))))
            else:
                current_qty = Decimal("0")

            if quantity > current_qty:
                logger.warning(
                    f"Reduce-only: clamping order from {quantity} to {current_qty} "
                    f"to prevent position flip for {symbol}"
                )
                quantity = current_qty
                trace.quantity = quantity
                reduce_only_clamped = True

            if quantity <= 0:
                logger.info(f"Reduce-only: no position to reduce for {symbol}, skipping order")
                trace.record_outcome(
                    "reduce_only",
                    "blocked",
                    detail="reduce_only_empty_position",
                )
                self._order_submitter.record_rejection(
                    symbol, side.value, quantity, price, "reduce_only_empty_position"
                )
                return quantity, self._finalize_decision_trace(
                    trace,
                    status=OrderSubmissionStatus.BLOCKED,
                    reason="reduce_only_empty_position",
                )

        order_for_check = {
            "symbol": symbol,
            "side": side.value,
            "quantity": float(quantity),
            "reduce_only": reduce_only_flag,
        }

        if not risk_manager.check_order(order_for_check):
            error_msg = (
                f"Order blocked by risk manager: "
                f"reduce_only_mode={reduce_only_mode}, "
                f"daily_pnl_triggered={daily_pnl_triggered}"
            )
            logger.warning(error_msg)
            await self._notify(
                title="Order Blocked - Reduce Only Mode",
                message=f"Cannot open new {side.value} position for {symbol} while in reduce-only mode",
                severity=AlertSeverity.WARNING,
                context=order_for_check,
            )
            trace.record_outcome("reduce_only", "blocked", detail=error_msg)
            self._order_submitter.record_rejection(
                symbol, side.value, quantity, price, "reduce_only_mode_blocked"
            )
            return quantity, self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason=error_msg,
            )
        trace.record_outcome(
            "reduce_only",
            "passed",
            detail="clamped" if reduce_only_clamped else None,
        )
        return quantity, None

    async def _check_mark_staleness(
        self,
        *,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        price: Decimal,
        reduce_only_flag: bool,
        trace: OrderDecisionTrace,
    ) -> OrderSubmissionResult | None:
        if self.context.risk_manager is None:
            trace.record_outcome("mark_staleness", "skipped")
            return None

        if self.context.risk_manager.check_mark_staleness(symbol):
            config = self.context.risk_manager.config
            if config is not None:
                allow_reduce = config.mark_staleness_allow_reduce_only
                cooldown = config.mark_staleness_cooldown_seconds
                self._append_event(
                    "stale_mark_detected",
                    {
                        "symbol": symbol,
                        "side": side.value,
                        "allowed_reduce_only": allow_reduce and reduce_only_flag,
                        "timestamp": time.time(),
                    },
                )
                self._degradation.pause_symbol(
                    symbol=symbol,
                    seconds=cooldown,
                    reason="mark_staleness",
                    allow_reduce_only=allow_reduce,
                )
                if allow_reduce and reduce_only_flag:
                    logger.info(
                        f"Mark stale for {symbol} but allowing reduce-only order",
                        operation="degradation",
                    )
                    trace.record_outcome(
                        "mark_staleness",
                        "allowed",
                        detail="reduce_only",
                    )
                    return None

                logger.warning(f"Order blocked: mark price stale for {symbol}")
                self._order_submitter.record_rejection(
                    symbol, side.value, quantity, price, "mark_staleness"
                )
                await self._notify(
                    title="Order Blocked - Stale Mark Price",
                    message=f"Cannot place order for {symbol}: mark price data is stale",
                    severity=AlertSeverity.WARNING,
                    context={"symbol": symbol, "side": side.value},
                )
                trace.record_outcome("mark_staleness", "blocked", detail="stale")
                return self._finalize_decision_trace(
                    trace,
                    status=OrderSubmissionStatus.BLOCKED,
                    reason="mark_staleness",
                )

            logger.warning(f"Order blocked: mark price stale for {symbol}")
            self._append_event(
                "stale_mark_detected",
                {
                    "symbol": symbol,
                    "side": side.value,
                    "allowed_reduce_only": False,
                    "timestamp": time.time(),
                },
            )
            await self._notify(
                title="Order Blocked - Stale Mark Price",
                message=f"Cannot place order for {symbol}: mark price data is stale",
                severity=AlertSeverity.WARNING,
                context={"symbol": symbol, "side": side.value},
            )
            trace.record_outcome("mark_staleness", "blocked", detail="stale")
            return self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason="mark_staleness",
            )

        trace.record_outcome("mark_staleness", "passed")
        return None

    async def _run_order_validator_guards(
        self,
        *,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        equity: Decimal,
        quantity: Decimal,
        reduce_only_flag: bool,
        trace: OrderDecisionTrace,
    ) -> tuple[Decimal, Decimal, bool, OrderSubmissionResult | None]:
        effective_price = price
        if self._order_validator is None:
            trace.record_outcome("order_validation", "skipped")
            return quantity, effective_price, reduce_only_flag, None

        try:
            with profile_span("pre_trade_validation", {"symbol": symbol}) as _val_span:
                product = self._state_collector.require_product(symbol, product=None)
                effective_price = self._state_collector.resolve_effective_price(
                    symbol, side.value.lower(), price, product
                )

                try:
                    quantity, _ = self._order_validator.validate_exchange_rules(
                        symbol=symbol,
                        side=side,
                        order_type=OrderType.MARKET,
                        order_quantity=quantity,
                        price=None,
                        effective_price=effective_price,
                        product=product,
                    )
                    trace.quantity = quantity
                    trace.record_outcome("exchange_rules", "passed")
                except ValidationError as exc:
                    trace.record_outcome("exchange_rules", "blocked", detail=str(exc))
                    raise

                current_positions_dict = self._state_collector.build_positions_dict(
                    list(self._current_positions.values())
                )
                try:
                    self._order_validator.run_pre_trade_validation(
                        symbol=symbol,
                        side=side,
                        order_quantity=quantity,
                        effective_price=effective_price,
                        product=product,
                        equity=equity,
                        current_positions=current_positions_dict,
                    )
                    trace.record_outcome("pre_trade_validation", "passed")
                except ValidationError as exc:
                    trace.record_outcome("pre_trade_validation", "blocked", detail=str(exc))
                    raise

                try:
                    self._order_validator.enforce_slippage_guard(
                        symbol, side, quantity, effective_price
                    )
                    trace.record_outcome("slippage_guard", "passed")
                    self._degradation.reset_slippage_failures(symbol)
                except ValidationError as slippage_exc:
                    trace.record_outcome(
                        "slippage_guard",
                        "blocked",
                        detail=str(slippage_exc),
                    )
                    config = self.context.risk_manager.config if self.context.risk_manager else None
                    if config is not None:
                        self._degradation.record_slippage_failure(symbol, config)
                    raise slippage_exc

                # Use container's tracker (validated at init, asserted non-None here)
                assert self.context.container is not None
                failure_tracker = self.context.container.validation_failure_tracker
                config = self.context.risk_manager.config if self.context.risk_manager else None
                preview_disable_threshold = config.preview_failure_disable_after if config else 5

                if (
                    self._order_validator.enable_order_preview
                    and failure_tracker.get_failure_count("order_preview")
                    >= preview_disable_threshold
                ):
                    logger.warning(
                        "Auto-disabling order preview due to repeated failures",
                        consecutive_failures=failure_tracker.get_failure_count("order_preview"),
                        threshold=preview_disable_threshold,
                        operation="degradation",
                        stage="preview_disable",
                    )
                    self._order_validator.enable_order_preview = False

                if self._order_validator.enable_order_preview:
                    try:
                        await self._order_validator.maybe_preview_order_async(
                            symbol=symbol,
                            side=side,
                            order_type=OrderType.MARKET,
                            order_quantity=quantity,
                            effective_price=effective_price,
                            stop_price=None,
                            tif=self.context.config.time_in_force,
                            reduce_only=reduce_only_flag,
                            leverage=None,
                        )
                        trace.record_outcome("order_preview", "passed")
                    except ValidationError as exc:
                        trace.record_outcome(
                            "order_preview",
                            "blocked",
                            detail=str(exc),
                        )
                        raise
                else:
                    trace.record_outcome("order_preview", "skipped")

                reduce_only_flag = self._order_validator.finalize_reduce_only_flag(
                    reduce_only_flag, symbol
                )
                trace.reduce_only_final = reduce_only_flag
        except ValidationError as exc:
            logger.warning(f"Pre-trade guard rejected order: {exc}")
            blocked_stage = None
            for stage, outcome in trace.outcomes.items():
                if outcome.get("status") == "blocked":
                    blocked_stage = stage
                    break
            reason_code = blocked_stage or "order_validation"
            self._order_submitter.record_rejection(
                symbol, side.value, quantity, effective_price, reason_code
            )
            await self._notify(
                title="Order Blocked - Guard Rejection",
                message=f"Cannot place order for {symbol}: {exc}",
                severity=AlertSeverity.WARNING,
                context={"symbol": symbol, "side": side.value, "reason": str(exc)},
            )
            trace.record_outcome("order_validation", "blocked", detail=str(exc))
            return (
                quantity,
                effective_price,
                reduce_only_flag,
                self._finalize_decision_trace(
                    trace,
                    status=OrderSubmissionStatus.BLOCKED,
                    reason=str(exc),
                ),
            )
        except Exception as exc:
            logger.error(f"Guard check error: {exc}")
            self._order_submitter.record_rejection(
                symbol, side.value, quantity, price, "guard_error"
            )
            await self._notify(
                title="Order Blocked - Guard Error",
                message=f"Cannot place order for {symbol}: guard check failed",
                severity=AlertSeverity.ERROR,
                context={"symbol": symbol, "side": side.value, "error": str(exc)},
            )
            trace.record_outcome("order_validation", "error", detail=str(exc))
            return (
                quantity,
                effective_price,
                reduce_only_flag,
                self._finalize_decision_trace(
                    trace,
                    status=OrderSubmissionStatus.FAILED,
                    error=str(exc),
                ),
            )

        return quantity, effective_price, reduce_only_flag, None

    async def _validate_and_place_order(
        self,
        symbol: str,
        decision: Decision,
        price: Decimal,
        equity: Decimal,
        quantity_override: Decimal | None = None,
        reduce_only_requested: bool = False,
    ) -> OrderSubmissionResult:
        """Validate and submit an order through the guard stack.

        Returns:
            OrderSubmissionResult describing success/blocked/failed.
        """
        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL

        # Early check: is this order actually reduce-only? (needed for degradation check)
        current_pos = self._current_positions.get(symbol)
        is_reducing = self._is_reduce_only_order(current_pos, side)
        reduce_only_flag = is_reducing

        decision_id = self._order_submitter.generate_client_order_id(None)
        trace = OrderDecisionTrace(
            symbol=symbol,
            side=side.value,
            price=price,
            equity=equity,
            quantity=None,
            reduce_only=reduce_only_requested,
            reduce_only_final=reduce_only_flag,
            reason=decision.reason,
            decision_id=decision_id,
            bot_id=str(self.context.bot_id) if self.context.bot_id is not None else None,
        )

        config = getattr(self.context.risk_manager, "config", None)
        kill_switch_enabled = getattr(config, "kill_switch_enabled", False) is True
        if kill_switch_enabled:
            trace.record_outcome(
                "kill_switch",
                "blocked",
                detail="kill_switch_enabled",
            )
            self._order_submitter.record_rejection(
                symbol,
                side.value,
                Decimal("0"),
                price,
                "kill_switch",
                client_order_id=decision_id,
            )
            return self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.BLOCKED,
                reason="kill_switch",
            )

        result = await self._check_degradation_gate(
            symbol=symbol,
            side=side,
            price=price,
            trace=trace,
            reduce_only_flag=reduce_only_flag,
        )
        if result is not None:
            return result

        quantity, result = self._calculate_quantity_and_record(
            symbol=symbol,
            side=side,
            price=price,
            equity=equity,
            quantity_override=quantity_override,
            trace=trace,
        )
        if result is not None:
            return result

        result = await self._check_reduce_only_request(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            reduce_only_requested=reduce_only_requested,
            is_reducing=is_reducing,
            trace=trace,
        )
        if result is not None:
            return result

        result = await self._run_security_validation(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            equity=equity,
            trace=trace,
        )
        if result is not None:
            return result

        quantity, result = await self._apply_reduce_only_mode(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            reduce_only_flag=reduce_only_flag,
            is_reducing=is_reducing,
            current_pos=current_pos,
            trace=trace,
        )
        if result is not None:
            return result

        result = await self._check_mark_staleness(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            reduce_only_flag=reduce_only_flag,
            trace=trace,
        )
        if result is not None:
            return result

        (
            quantity,
            effective_price,
            reduce_only_flag,
            result,
        ) = await self._run_order_validator_guards(
            symbol=symbol,
            side=side,
            price=price,
            equity=equity,
            quantity=quantity,
            reduce_only_flag=reduce_only_flag,
            trace=trace,
        )
        if result is not None:
            return result

        # Place order via OrderSubmitter for proper ID tracking and telemetry
        submission_outcome = await self._broker_calls(
            self._order_submitter.submit_order_with_result,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            order_quantity=quantity,
            price=None,  # Market order
            effective_price=effective_price,
            stop_price=None,
            tif=self.context.config.time_in_force,
            reduce_only=reduce_only_flag,
            leverage=None,
            client_order_id=decision_id,
        )

        # Notify on successful order placement
        if submission_outcome.success:
            order_id = submission_outcome.order_id
            trace.order_id = order_id
            await self._notify(
                title="Order Executed",
                message=f"{side.value} {quantity} {symbol} at ~{price}",
                severity=AlertSeverity.INFO,
                context={
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                    "price": str(price),
                    "order_id": order_id,
                },
            )

            # Record trade in status reporter
            self._status_reporter.add_trade(
                {
                    "symbol": symbol,
                    "side": side.value,
                    "quantity": str(quantity),
                    "price": str(price),
                    "order_id": order_id,
                }
            )
            trace.record_outcome("submit_order", "success", order_id=order_id)
            return self._finalize_decision_trace(
                trace,
                status=OrderSubmissionStatus.SUCCESS,
                order_id=order_id,
            )
        reason = submission_outcome.reason or "broker_rejected"
        logger.warning(
            "Order submission failed",
            symbol=symbol,
            side=side.value,
            reason=reason,
            reason_detail=submission_outcome.reason_detail,
            operation="order_submit",
            stage="failed",
        )
        trace.record_outcome(
            "submit_order",
            "failed",
            detail=reason,
            reason_detail=submission_outcome.reason_detail,
            error=submission_outcome.error,
        )
        return self._finalize_decision_trace(
            trace,
            status=OrderSubmissionStatus.FAILED,
            reason=reason,
            error=submission_outcome.error_message,
        )

    def reset_daily_tracking(self) -> None:
        """Reset daily PnL tracking and guard cache (start of trading day)."""
        try:
            broker = self.context.broker
            if broker is None:
                logger.warning(
                    "Cannot reset daily tracking without broker",
                    operation="daily_tracking",
                    stage="missing_broker",
                )
                return

            balances = broker.list_balances()
            equity, _, _ = self._state_collector.calculate_equity_from_balances(balances)

            if self.context.risk_manager is not None:
                self.context.risk_manager.reset_daily_tracking()

            if self._guard_manager is not None:
                self._guard_manager.invalidate_cache()

            logger.info(
                "Daily tracking reset",
                operation="daily_tracking",
                stage="reset",
                equity=float(equity),
            )
        except Exception as exc:
            logger.error(
                "Failed to reset daily tracking",
                error_message=str(exc),
                operation="daily_tracking",
                stage="reset",
            )

    # =========================================================================
    # PUBLIC SUBMISSION ENTRYPOINT
    # =========================================================================
    # This is the canonical order submission path. All order execution should
    # route through this method to ensure the full guard stack is applied:
    # degradation gate → sizing → security → risk → staleness → validator
    # =========================================================================

    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        price: Decimal,
        equity: Decimal,
        *,
        quantity_override: Decimal | None = None,
        reduce_only: bool = False,
        reason: str = "external_submission",
        confidence: float = 1.0,
    ) -> OrderSubmissionResult:
        """Public entrypoint for order submission through the canonical guard stack.

        This method provides external callers (OrderRouter, TUI actions, etc.) access
        to the full pre-trade validation pipeline. All orders should route through
        here to ensure consistent guard enforcement.

        Args:
            symbol: Trading symbol (e.g., "BTC-USD").
            side: Order side (BUY or SELL).
            price: Current market price for validation.
            equity: Current account equity for position sizing.
            quantity_override: If provided, uses this quantity instead of dynamic sizing.
            reduce_only: If True, order is reduce-only (affects degradation gate).
            reason: Reason for the order (for logging/telemetry).
            confidence: Decision confidence score (0.0-1.0).

        Note:
            This method delegates to _validate_and_place_order after constructing
            a Decision object. The full guard stack is applied.

        Returns:
            OrderSubmissionResult describing the outcome.
        """
        # Construct Decision from inputs
        action = Action.BUY if side == OrderSide.BUY else Action.SELL
        decision = Decision(
            action=action,
            reason=reason,
            confidence=confidence,
        )

        # Pass quantity override through to guard stack sizing.
        return await self._validate_and_place_order(
            symbol,
            decision,
            price,
            equity,
            quantity_override=quantity_override,
            reduce_only_requested=reduce_only,
        )

    async def shutdown(self) -> None:
        self._transition_state(EngineState.STOPPING, reason="shutdown_called")

        # Stop WS health watchdog
        if self._ws_health_task is not None and not self._ws_health_task.done():
            self._ws_health_task.cancel()
            try:
                await self._ws_health_task
            except asyncio.CancelledError:
                pass
            self._ws_health_task = None
            logger.info(
                "Stopped WS health watchdog",
                operation="ws_health",
                stage="stop",
            )

        # Stop streaming
        stop_streaming_background(self)
        logger.info(
            "Stopped WebSocket streaming",
            operation="streaming",
            stage="stop",
        )

        # Stop health check runner
        await self._health_check_runner.stop()

        await self._system_maintenance.stop()
        await self._status_reporter.stop()
        await self._heartbeat.stop()
        shutdown = getattr(self._broker_calls, "shutdown", None)
        if callable(shutdown):
            shutdown()
        await super().shutdown()
        self._transition_state(EngineState.STOPPED, reason="shutdown_complete")

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)

    async def _audit_orders(self) -> None:
        """Audit open orders for reconciliation."""
        broker = self.context.broker
        if broker is None:
            return
        try:
            # Fetch open orders
            response: Any = None
            module_name = getattr(broker, "__module__", "")
            if "coinbase" in module_name:
                # Prefer Coinbase client list_orders to get the raw dict shape expected by StatusReporter.
                client = getattr(broker, "client", None)
                client_list_orders = getattr(client, "list_orders", None)
                if callable(client_list_orders):
                    response = await self._broker_calls(client_list_orders, order_status="OPEN")
            if response is None:
                list_orders = getattr(broker, "list_orders", None)
                if callable(list_orders):
                    try:
                        response = await self._broker_calls(list_orders, order_status="OPEN")
                    except TypeError:
                        response = await self._broker_calls(list_orders, status=["OPEN"])

            orders: list[Any]
            if isinstance(response, dict):
                orders = list(response.get("orders", []))
            elif isinstance(response, list):
                orders = list(response)
            else:
                orders = []

            await self._reconcile_open_orders(orders)

            if orders:
                logger.info(
                    "AUDIT: Found OPEN orders",
                    open_order_count=len(orders),
                    operation="order_audit",
                    stage="list",
                )

            self._record_unfilled_order_alerts(orders)

            # Update status reporter (expects dict-like orders).
            status_orders: list[dict[str, Any]] = []
            for order in orders:
                if isinstance(order, dict):
                    status_orders.append(order)
                    continue
                status_orders.append(
                    {
                        "order_id": self._get_order_field(order, "order_id", "id") or "",
                        "product_id": self._get_order_field(order, "product_id", "symbol") or "",
                        "side": self._get_order_field(order, "side") or "",
                        "status": self._get_order_field(order, "status") or "",
                        "price": self._get_order_field(order, "price"),
                        "size": self._get_order_field(order, "size", "quantity"),
                        "created_time": self._get_order_field(order, "created_time", "created_at"),
                        "filled_size": self._get_order_field(
                            order, "filled_size", "filled_quantity"
                        ),
                        "average_filled_price": self._get_order_field(
                            order, "average_filled_price", "avg_fill_price"
                        ),
                    }
                )
            self._status_reporter.update_orders(status_orders)

            # Update Account Metrics (every 60 cycles ~ 1 minute)
            if self._cycle_count % 60 == 0:
                try:
                    balances = await self._broker_calls(broker.list_balances)
                    # Check if broker supports transaction summary (Coinbase specific)
                    summary = {}
                    if hasattr(broker, "client") and hasattr(
                        broker.client, "get_transaction_summary"
                    ):
                        try:
                            summary = await self._broker_calls(
                                broker.client.get_transaction_summary
                            )
                        except Exception:
                            pass  # Feature might not be available or API mode issue

                    self._status_reporter.update_account(balances, summary)
                except Exception as e:
                    logger.warning(f"Failed to update account metrics: {e}")

        except Exception as e:
            logger.warning(f"Failed to audit orders: {e}")
