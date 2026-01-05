"""
Telemetry and health monitoring for live trading coordinators.

This module provides utilities for real-time market data processing and system health:

Functions
---------
- ``extract_mark_from_message``: Extract mark price from WebSocket messages
- ``update_mark_and_metrics``: Update mark price windows and risk manager state
- ``health_check``: Check coordinator health status

Mark Price Extraction
---------------------
The ``extract_mark_from_message`` function handles various WebSocket message formats:

1. If ``best_bid`` and ``best_ask`` are present, calculates mid-price
2. Falls back to ``last`` or ``price`` fields
3. Returns ``None`` for invalid/missing data

Health Checks
-------------
The ``health_check`` function validates:

- Account telemetry service availability
- Market monitor connectivity
- WebSocket stream status
- Background task health

Integration
-----------
These utilities are used by the ``TelemetryCoordinator`` and ``PerpsCoordinator`` engines
to maintain real-time market state and detect connectivity issues.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities import utc_now
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from gpt_trader.features.live_trade.engines.base import CoordinatorContext

logger = get_logger(__name__, component="telemetry_health")


def extract_mark_from_message(msg: dict[str, Any]) -> Decimal | None:
    bid = msg.get("best_bid") or msg.get("bid")
    ask = msg.get("best_ask") or msg.get("ask")
    try:
        if bid is not None and ask is not None:
            mark = (Decimal(str(bid)) + Decimal(str(ask))) / Decimal("2")
            return mark if mark > 0 else None
        raw_mark = msg.get("last") or msg.get("price")
        if raw_mark is not None:
            mark = Decimal(str(raw_mark))
            return mark if mark > 0 else None
    except Exception as exc:
        logger.error(
            "Failed to extract mark from message",
            error_type=type(exc).__name__,
            error_message=str(exc),
            operation="extract_mark",
            bid=str(bid) if bid else None,
            ask=str(ask) if ask else None,
        )
        return None
    return None


def update_mark_and_metrics(
    coordinator: Any,  # Accepts various coordinator types
    ctx: CoordinatorContext,
    symbol: str,
    mark: Decimal,
) -> None:
    strategy_coordinator = getattr(ctx, "strategy_coordinator", None)
    if strategy_coordinator and hasattr(strategy_coordinator, "update_mark_window"):
        try:
            strategy_coordinator.update_mark_window(symbol, mark)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to update mark window",
                error=str(exc),
                symbol=symbol,
                exc_info=True,
                operation="telemetry_stream",
                stage="mark_window",
            )
    else:
        runtime_state = ctx.runtime_state
        if runtime_state is not None:
            with runtime_state.mark_lock:
                window = runtime_state.mark_windows.setdefault(symbol, [])
                window.append(mark)
                max_size = max(ctx.config.short_ma, ctx.config.long_ma) + 5
                if len(window) > max_size:
                    runtime_state.mark_windows[symbol] = window[-max_size:]

    extras = getattr(ctx.registry, "extras", {})
    monitor = None
    if isinstance(extras, dict):
        monitor = extras.get("market_monitor")
    if monitor is None:
        monitor = coordinator._market_monitor
        if monitor is not None:
            try:
                monitor.record_update(symbol)
            except Exception:  # pragma: no cover - defensive logging
                logger.debug(
                    "Failed to record market update",
                    symbol=symbol,
                    exc_info=True,
                    operation="telemetry_stream",
                    stage="market_monitor",
                )

    risk_manager = ctx.risk_manager
    if risk_manager is not None:
        timestamp = utc_now()
        stored = timestamp
        record_fn = getattr(risk_manager, "record_mark_update", None)
        if callable(record_fn):
            try:
                result = record_fn(symbol, timestamp)
                if result is not None:
                    stored = result
            except Exception:  # pragma: no cover - defensive logging
                logger.exception(
                    "WS mark update bookkeeping failed",
                    symbol=symbol,
                    operation="telemetry_stream",
                    stage="risk_update",
                )
                stored = timestamp

        try:
            last_updates = getattr(risk_manager, "last_mark_update", None)
            if not isinstance(last_updates, dict):
                last_updates = {}
                setattr(risk_manager, "last_mark_update", last_updates)
            last_updates[symbol] = stored
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "Failed to persist last mark update",
                symbol=symbol,
                operation="telemetry_stream",
                stage="risk_update_persist",
                exc_info=True,
            )

    # _emit_metric(
    #     ctx.event_store,
    #     ctx.bot_id,
    #     {"event_type": "ws_mark_update", "symbol": symbol, "mark": str(mark)},
    # )
    logger.debug(
        "Emitting metric (placeholder)",
        event_type="ws_mark_update",
        symbol=symbol,
        mark=str(mark),
        bot_id=ctx.bot_id,
    )


def health_check(
    coordinator: Any,
) -> Any:  # Returns HealthStatus but accepts various coordinator types
    from gpt_trader.features.live_trade.engines.base import HealthStatus

    raw_extras = getattr(coordinator.context.registry, "extras", {})
    if not isinstance(raw_extras, dict):
        try:
            raw_extras = dict(raw_extras)
        except Exception as exc:
            logger.error(
                "Failed to convert raw_extras to dict",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="health_check",
                extras_type=type(raw_extras).__name__,
            )
            raw_extras = {}
    account_telemetry = raw_extras.get("account_telemetry")
    healthy = account_telemetry is not None
    details = {
        "has_account_telemetry": account_telemetry is not None,
        "has_market_monitor": raw_extras.get("market_monitor") is not None
        or coordinator._market_monitor is not None,
        "streaming_active": coordinator._stream_task is not None
        and not coordinator._stream_task.done(),
        "background_tasks": len(coordinator._background_tasks),
    }
    return HealthStatus(healthy=healthy, component=coordinator.name, details=details)


def update_orderbook_snapshot(
    ctx: CoordinatorContext,
    event: Any,
) -> None:
    """
    Update orderbook snapshot from WebSocket OrderbookUpdate event.

    Stores the latest depth snapshot in runtime_state for strategy consumption.

    Args:
        ctx: Coordinator context with runtime_state
        event: OrderbookUpdate dataclass from ws_events.py
    """
    runtime_state = ctx.runtime_state
    if runtime_state is None:
        return

    # Check if runtime_state has orderbook support
    if not hasattr(runtime_state, "orderbook_lock") or not hasattr(
        runtime_state, "orderbook_snapshots"
    ):
        return

    try:
        from gpt_trader.features.brokerages.coinbase.market_data_features import DepthSnapshot

        snapshot = DepthSnapshot.from_orderbook_update(event)
        product_id = getattr(event, "product_id", None)

        if product_id:
            with runtime_state.orderbook_lock:
                runtime_state.orderbook_snapshots[product_id] = snapshot

            logger.debug(
                "Updated orderbook snapshot",
                symbol=product_id,
                spread_bps=snapshot.spread_bps,
                bid_levels=len(snapshot.bids),
                ask_levels=len(snapshot.asks),
                operation="telemetry_orderbook",
            )

            # Emit to EventStore (throttled)
            emit_orderbook_snapshot(ctx, product_id)
    except Exception as exc:
        logger.error(
            "Failed to update orderbook snapshot",
            error_type=type(exc).__name__,
            error_message=str(exc),
            operation="telemetry_orderbook",
            exc_info=True,
        )


def update_trade_aggregator(
    ctx: CoordinatorContext,
    event: Any,
) -> None:
    """
    Update trade aggregator from WebSocket TradeEvent.

    Maintains rolling trade flow statistics for volume analysis.

    Args:
        ctx: Coordinator context with runtime_state
        event: TradeEvent dataclass from ws_events.py
    """
    runtime_state = ctx.runtime_state
    if runtime_state is None:
        return

    # Check if runtime_state has trade aggregator support
    if not hasattr(runtime_state, "trade_lock") or not hasattr(
        runtime_state, "trade_aggregators"
    ):
        return

    try:
        from gpt_trader.features.brokerages.coinbase.market_data_features import TradeTapeAgg

        product_id = getattr(event, "product_id", None)
        price = getattr(event, "price", None)
        size = getattr(event, "size", None)
        side = getattr(event, "side", None)
        timestamp = getattr(event, "timestamp", None)

        if not all([product_id, price is not None, size is not None, side]):
            return

        with runtime_state.trade_lock:
            # Create aggregator if it doesn't exist (60-second rolling window)
            if product_id not in runtime_state.trade_aggregators:
                runtime_state.trade_aggregators[product_id] = TradeTapeAgg(duration_seconds=60)

            agg = runtime_state.trade_aggregators[product_id]
            agg.add_trade(price, size, side, timestamp)

        logger.debug(
            "Updated trade aggregator",
            symbol=product_id,
            price=str(price),
            size=str(size),
            side=side,
            operation="telemetry_trade",
        )

        # Emit to EventStore (throttled)
        emit_trade_flow_summary(ctx, product_id)
    except Exception as exc:
        logger.error(
            "Failed to update trade aggregator",
            error_type=type(exc).__name__,
            error_message=str(exc),
            operation="telemetry_trade",
            exc_info=True,
        )


# Snapshot throttling state (per-symbol)
_last_snapshot_times: dict[str, float] = {}
_SNAPSHOT_INTERVAL_SECONDS = 5.0  # Emit snapshots every 5 seconds per symbol


def emit_orderbook_snapshot(
    ctx: "CoordinatorContext",
    symbol: str,
) -> None:
    """
    Emit orderbook snapshot to EventStore for backtesting.

    This function is called periodically (throttled) to persist orderbook
    depth data for historical analysis and strategy backtesting.

    Args:
        ctx: Coordinator context with event_store and runtime_state
        symbol: Product symbol to emit snapshot for
    """
    import time

    runtime_state = ctx.runtime_state
    event_store = ctx.event_store

    if runtime_state is None or event_store is None:
        return

    if not hasattr(runtime_state, "orderbook_lock") or not hasattr(
        runtime_state, "orderbook_snapshots"
    ):
        return

    # Throttle snapshots
    current_time = time.time()
    last_time = _last_snapshot_times.get(f"orderbook:{symbol}", 0.0)
    if current_time - last_time < _SNAPSHOT_INTERVAL_SECONDS:
        return

    try:
        with runtime_state.orderbook_lock:
            snapshot = runtime_state.orderbook_snapshots.get(symbol)
            if snapshot is None:
                return

            # Extract relevant data
            bid_depth, ask_depth = snapshot.get_depth(10)
            event_data = {
                "symbol": symbol,
                "spread_bps": snapshot.spread_bps,
                "mid_price": str(snapshot.mid) if snapshot.mid else None,
                "bid_depth_l10": str(bid_depth),
                "ask_depth_l10": str(ask_depth),
                "bid_levels": len(snapshot.bids),
                "ask_levels": len(snapshot.asks),
                "bot_id": ctx.bot_id,
            }

        event_store.append("orderbook_snapshot", event_data)
        _last_snapshot_times[f"orderbook:{symbol}"] = current_time

        logger.debug(
            "Emitted orderbook snapshot",
            symbol=symbol,
            spread_bps=snapshot.spread_bps,
            operation="event_store",
        )
    except Exception as exc:
        logger.error(
            "Failed to emit orderbook snapshot",
            error_type=type(exc).__name__,
            error_message=str(exc),
            symbol=symbol,
            operation="event_store",
            exc_info=True,
        )


def emit_trade_flow_summary(
    ctx: "CoordinatorContext",
    symbol: str,
) -> None:
    """
    Emit trade flow summary to EventStore for backtesting.

    This function is called periodically (throttled) to persist trade flow
    statistics for historical analysis and strategy backtesting.

    Args:
        ctx: Coordinator context with event_store and runtime_state
        symbol: Product symbol to emit summary for
    """
    import time

    runtime_state = ctx.runtime_state
    event_store = ctx.event_store

    if runtime_state is None or event_store is None:
        return

    if not hasattr(runtime_state, "trade_lock") or not hasattr(
        runtime_state, "trade_aggregators"
    ):
        return

    # Throttle snapshots
    current_time = time.time()
    last_time = _last_snapshot_times.get(f"trade:{symbol}", 0.0)
    if current_time - last_time < _SNAPSHOT_INTERVAL_SECONDS:
        return

    try:
        with runtime_state.trade_lock:
            agg = runtime_state.trade_aggregators.get(symbol)
            if agg is None:
                return

            stats = agg.get_stats()

        event_data = {
            "symbol": symbol,
            "trade_count": stats.get("count", 0),
            "volume": str(stats.get("volume", 0)),
            "vwap": str(stats.get("vwap", 0)),
            "avg_size": str(stats.get("avg_size", 0)),
            "aggressor_ratio": stats.get("aggressor_ratio", 0.0),
            "bot_id": ctx.bot_id,
        }

        event_store.append("trade_flow_summary", event_data)
        _last_snapshot_times[f"trade:{symbol}"] = current_time

        logger.debug(
            "Emitted trade flow summary",
            symbol=symbol,
            trade_count=stats.get("count", 0),
            operation="event_store",
        )
    except Exception as exc:
        logger.error(
            "Failed to emit trade flow summary",
            error_type=type(exc).__name__,
            error_message=str(exc),
            symbol=symbol,
            operation="event_store",
            exc_info=True,
        )


__all__ = [
    "extract_mark_from_message",
    "update_mark_and_metrics",
    "update_orderbook_snapshot",
    "update_trade_aggregator",
    "emit_orderbook_snapshot",
    "emit_trade_flow_summary",
    "health_check",
]
