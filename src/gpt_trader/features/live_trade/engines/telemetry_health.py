from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from gpt_trader.utilities import utc_now
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from gpt_trader.features.live_trade.engines.base import CoordinatorContext

    # TelemetryEngine is a missing import. Keeping it here for reference until it's located.
    # from gpt_trader.orchestration.engines.telemetry_coordinator import TelemetryEngine

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
    coordinator: Any,  # Changed from "TelemetryEngine" to Any due to missing import
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
) -> Any:  # Changed from "TelemetryEngine" to Any due to missing import
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


__all__ = [
    "extract_mark_from_message",
    "update_mark_and_metrics",
    "health_check",
]
