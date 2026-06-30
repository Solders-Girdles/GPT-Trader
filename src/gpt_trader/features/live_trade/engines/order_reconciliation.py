"""Order reconciliation for the live TradingEngine.

Startup rehydration and drift reconciliation between persisted open orders and
the broker's view (recover unknown bot orders, refresh missing persisted orders,
escalate on repeated drift). Extracted from strategy.py following the engine's
collaborator-function pattern; the engine instance is passed in and these
functions operate on its state and delegate back via thin engine methods.
"""

from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.persistence.orders_store import OrderRecord
from gpt_trader.persistence.orders_store import OrderStatus as PersistedOrderStatus
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="trading_engine")

# Consecutive order reconciliation drift detections required before triggering
# an escalation (pause + reduce-only). Conservative to avoid flapping on
# transient list_orders inconsistencies.
ORDER_RECONCILIATION_DRIFT_MAX_FAILURES = 3


def rehydrate_open_orders(engine: Any) -> None:
    """Restore open order IDs from the orders store after a restart."""
    if engine._orders_store is None:
        return
    bot_id = str(engine.context.bot_id or engine.context.config.profile or "")
    try:
        pending = engine._orders_store.get_pending_orders(bot_id=bot_id or None)
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

    engine._open_orders[:] = order_ids
    logger.info(
        "Rehydrated open orders from persistence",
        count=len(order_ids),
        operation="order_recovery",
    )


async def recover_unknown_bot_orders(
    engine: Any,
    orders: list[Any],
    *,
    bot_id: str,
) -> list[OrderRecord]:
    if not orders or engine._orders_store is None:
        return []
    now = datetime.now(timezone.utc)
    recovered: list[OrderRecord] = []
    for order in orders:
        record = engine._build_record_from_broker_order(order, bot_id=bot_id, now=now)
        if record is None:
            continue
        metadata = dict(record.metadata or {})
        metadata["note"] = "backfilled_from_broker"
        record = replace(record, metadata=metadata)
        try:
            await engine._broker_calls(engine._orders_store.upsert_by_client_id, record)
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
        if record.order_id not in engine._open_orders:
            engine._open_orders.append(record.order_id)
        recovered.append(record)

    if recovered:
        payload = {
            "timestamp": time.time(),
            "bot_id": bot_id,
            "recovered_order_ids": [record.order_id for record in recovered][:10],
            "recovered_order_count": len(recovered),
            "open_order_count": len(engine._open_orders),
        }
        engine._append_event("order_reconciliation_recovered", payload)
        logger.info(
            "Recovered unknown bot orders",
            recovered_order_count=len(recovered),
            operation="order_reconciliation",
            stage="recover",
        )

    return recovered


async def refresh_missing_persisted_orders(
    engine: Any,
    records: list[OrderRecord],
    *,
    bot_id: str,
) -> list[Any]:
    if not records or engine._orders_store is None:
        return []
    broker = engine.context.broker
    if broker is None:
        return []
    get_order = getattr(broker, "get_order", None)
    if not callable(get_order):
        return []
    now = datetime.now(timezone.utc)
    refreshed_orders: list[Any] = []
    for record in records:
        try:
            order = await engine._broker_calls(get_order, record.order_id)
            if order is None and record.client_order_id != record.order_id:
                order = await engine._broker_calls(get_order, record.client_order_id)
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
        update_record = engine._build_record_from_broker_order(order, bot_id=bot_id, now=now)
        if update_record is None:
            continue
        update_metadata = dict(update_record.metadata or {})
        update_metadata["note"] = "refresh_missing"
        metadata = engine._merge_metadata(record.metadata, update_metadata)
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
            await engine._broker_calls(engine._orders_store.upsert_by_client_id, updated)
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
            if updated.order_id not in engine._open_orders:
                engine._open_orders.append(updated.order_id)
            refreshed_orders.append(order)
        else:
            if updated.order_id in engine._open_orders:
                engine._open_orders.remove(updated.order_id)

    return refreshed_orders


async def reconcile_open_orders(engine: Any, orders: list[Any]) -> None:
    """Reconcile internal open-order tracking with broker + persistence state.

    This addresses a crash-recovery edge case: orders are persisted as pending with
    ``order_id == client_order_id == submit_id`` before the broker responds with
    the canonical broker order ID. After restart, rehydration can restore the
    submit_id into ``engine._open_orders``. On the next audit, we normalize tracked
    IDs back to broker order IDs when the broker returns both IDs.
    """
    bot_id = str(engine.context.bot_id or engine.context.config.profile or "live")
    prefix = f"{bot_id}_"

    pending: list[OrderRecord] = []
    pending_by_client_id: dict[str, OrderRecord] = {}
    tracked_ids: set[str] = {str(order_id) for order_id in engine._open_orders}

    if engine._orders_store is not None:
        try:
            pending = await engine._broker_calls(
                engine._orders_store.get_pending_orders,
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
            refreshed_orders = await engine._refresh_missing_persisted_orders(
                pending,
                bot_id=bot_id,
            )
        if refreshed_orders:
            orders = refreshed_orders
        else:
            if not pending and engine._open_orders:
                engine._open_orders.clear()
            engine._order_reconciliation_drift_failures = 0
            engine._order_reconciliation_drift_escalated = False
            return

    order_index: dict[str, Any] = {}

    def _index_order(order: Any) -> None:
        if order is None:
            return
        order_id = engine._get_order_field(order, "order_id", "id")
        client_order_id = engine._get_order_field(order, "client_order_id", "client_id")
        if order_id is not None:
            order_index[str(order_id)] = order
        if client_order_id is not None:
            order_index[str(client_order_id)] = order

    for order in orders:
        _index_order(order)

    # Update persisted records that are still using submit_id as order_id, now that
    # the broker has returned the canonical order_id.
    if engine._orders_store is not None and pending_by_client_id:
        now = datetime.now(timezone.utc)
        for order in orders:
            order_id = engine._get_order_field(order, "order_id", "id")
            client_order_id = engine._get_order_field(order, "client_order_id", "client_id")
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
                await engine._broker_calls(engine._orders_store.upsert_by_client_id, updated)
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
            if str(record.order_id) not in open_ids and str(record.client_order_id) not in open_ids
        ]
        refreshed_orders = await engine._refresh_missing_persisted_orders(
            missing_records,
            bot_id=bot_id,
        )
        for refreshed_order in refreshed_orders:
            _index_order(refreshed_order)

    # Reconcile the tracked IDs: replace submit_ids with canonical order_ids and drop
    # IDs that are no longer open at the broker.
    snapshot = list(engine._open_orders)
    snapshot_set = set(snapshot)
    reconciled: list[str] = []
    seen: set[str] = set()

    def _canonical_order_id(tracked_id: str) -> str | None:
        order = order_index.get(tracked_id)
        if order is None:
            return None
        oid = engine._get_order_field(order, "order_id", "id") or tracked_id
        return str(oid)

    for tracked_id in snapshot:
        canonical = _canonical_order_id(tracked_id)
        if canonical is None or canonical in seen:
            continue
        seen.add(canonical)
        reconciled.append(canonical)

    # Preserve any IDs appended while we were reconciling (e.g., order submission)
    # without losing them via slice assignment.
    for tracked_id in list(engine._open_orders):
        if tracked_id in snapshot_set:
            continue
        canonical = _canonical_order_id(tracked_id) or str(tracked_id)
        if canonical in seen:
            continue
        seen.add(canonical)
        reconciled.append(canonical)

    if reconciled != snapshot:
        engine._open_orders[:] = reconciled

    # Drift detection: bot-owned open orders present at the broker but not tracked in
    # the store/open-orders list can indicate "ghost" orders after a crash.
    unknown_bot_orders: list[str] = []
    unknown_orders: list[Any] = []
    for order in orders:
        client_order_id = engine._get_order_field(order, "client_order_id", "client_id")
        if client_order_id is None:
            continue
        client_id_str = str(client_order_id)
        if not client_id_str.startswith(prefix):
            continue
        order_id = engine._get_order_field(order, "order_id", "id")
        order_id_str = str(order_id) if order_id is not None else client_id_str
        if order_id_str in tracked_ids or client_id_str in tracked_ids:
            continue
        unknown_bot_orders.append(order_id_str)
        unknown_orders.append(order)

    recovered_records = await engine._recover_unknown_bot_orders(
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
        engine._order_reconciliation_drift_failures = 0
        engine._order_reconciliation_drift_escalated = False
        return

    # Record drift event for observability.
    payload = {
        "timestamp": time.time(),
        "bot_id": bot_id,
        "unknown_bot_order_ids": unknown_bot_orders[:10],
        "unknown_bot_order_count": len(unknown_bot_orders),
        "open_order_count": len(orders),
    }
    engine._append_event("order_reconciliation_drift", payload)
    logger.warning(
        "Order reconciliation drift detected",
        unknown_bot_order_count=len(unknown_bot_orders),
        open_order_count=len(orders),
        operation="order_reconciliation",
        stage="drift",
    )

    if engine._order_reconciliation_drift_escalated:
        return

    engine._order_reconciliation_drift_failures += 1
    if engine._order_reconciliation_drift_failures < ORDER_RECONCILIATION_DRIFT_MAX_FAILURES:
        return

    engine._order_reconciliation_drift_escalated = True
    await engine._handle_order_reconciliation_drift(unknown_bot_orders)


async def handle_order_reconciliation_drift(engine: Any, order_ids: list[str]) -> None:
    """Escalate persistent order-reconciliation drift to graceful degradation."""
    risk_manager = engine.context.risk_manager
    config = risk_manager.config if risk_manager else None

    cooldown_seconds = getattr(config, "api_health_cooldown_seconds", 300) if config else 300
    if risk_manager is not None:
        risk_manager.set_reduce_only_mode(True, reason="order_reconciliation_drift")

    broker = engine.context.broker
    if broker is not None:
        cancel_executor = BrokerExecutor(broker=broker)
        retry_enabled = getattr(engine.context.config, "order_submission_retries_enabled", False)
        for order_id in order_ids:
            try:
                await engine._broker_calls(
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

    engine._degradation.pause_all(
        seconds=cooldown_seconds,
        reason="order_reconciliation_drift",
        allow_reduce_only=True,
    )

    # Record a synthetic rejection for telemetry.
    try:
        engine._order_submitter.record_rejection(
            symbol="*",
            side="*",
            quantity=Decimal("0"),
            price=None,
            reason="order_reconciliation_drift",
        )
    except Exception:
        pass

    await engine._notify(
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
