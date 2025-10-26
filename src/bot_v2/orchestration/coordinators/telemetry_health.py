from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

from bot_v2.utilities import utc_now

if TYPE_CHECKING:  # pragma: no cover
    from bot_v2.orchestration.coordinators.base import CoordinatorContext


def _logger():
    from bot_v2.orchestration.coordinators import telemetry as telemetry_module

    return telemetry_module.logger


def _emit_metric(event_store, bot_id: str, payload: dict[str, Any]) -> None:
    from bot_v2.orchestration.coordinators import telemetry as telemetry_module

    telemetry_module.emit_metric(event_store, bot_id, payload)


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
    except Exception:
        return None
    return None


def update_mark_and_metrics(
    coordinator: "TelemetryCoordinator",
    ctx: "CoordinatorContext",
    symbol: str,
    mark: Decimal,
) -> None:
    strategy_coordinator = getattr(ctx, "strategy_coordinator", None)
    if strategy_coordinator and hasattr(strategy_coordinator, "update_mark_window"):
        try:
            strategy_coordinator.update_mark_window(symbol, mark)
        except Exception as exc:  # pragma: no cover - defensive logging
            _logger().debug(
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
                _logger().debug(
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
                _logger().exception(
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
            _logger().debug(
                "Failed to persist last mark update",
                symbol=symbol,
                operation="telemetry_stream",
                stage="risk_update_persist",
                exc_info=True,
            )

    _emit_metric(
        ctx.event_store,
        ctx.bot_id,
        {"event_type": "ws_mark_update", "symbol": symbol, "mark": str(mark)},
    )


def health_check(coordinator: "TelemetryCoordinator"):
    from bot_v2.orchestration.coordinators.base import HealthStatus

    raw_extras = getattr(coordinator.context.registry, "extras", {})
    if not isinstance(raw_extras, dict):
        try:
            raw_extras = dict(raw_extras)
        except Exception:
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
