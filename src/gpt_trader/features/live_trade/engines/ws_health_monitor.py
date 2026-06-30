"""WebSocket health watchdog for the live TradingEngine.

Monitors WebSocket message/heartbeat staleness, drives reconnect/backoff, and
escalates to degradation when the feed goes stale. Extracted from strategy.py
following the engine's collaborator-function pattern; the engine keeps a thin
delegating method.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.engines.strategy import TradingEngine

logger = get_logger(__name__, component="trading_engine")


async def monitor_ws_health(engine: TradingEngine) -> None:
    """Monitor WebSocket health and trigger degradation on staleness.

    Periodically polls WS health metrics from the broker. If messages
    or heartbeats are stale beyond configured thresholds, triggers:
    - Reduce-only mode for affected symbols
    - Symbol pause for configured cooldown
    - Notification alerts

    On reconnect, pauses briefly to allow state synchronization.
    """
    risk_manager = engine.context.risk_manager
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

    while engine.running:
        try:
            # Get WS health from broker (if it supports the method)
            broker = engine.context.broker
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
                if isinstance(gap_count_raw, (int, float)) and not isinstance(gap_count_raw, bool)
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
                engine._append_event(
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
                engine._ws_reconnect_attempts = 0
                engine._ws_reconnect_delay = 1.0

                if engine._user_event_handler is not None:
                    backfill = getattr(engine._user_event_handler, "request_backfill", None)
                    if callable(backfill):
                        backfill(reason="ws_reconnect", run_in_thread=True)

                # Pause all symbols briefly after reconnect
                engine._degradation.pause_all(
                    seconds=reconnect_pause,
                    reason="ws_reconnect",
                    allow_reduce_only=True,
                )

                await engine._notify(
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
                engine._degradation.pause_all(
                    seconds=cooldown,
                    reason=stale_reason,
                    allow_reduce_only=True,
                )

                await engine._notify(
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
            if gap_count > 0 and engine._cycle_count % 60 == 0:
                logger.info(
                    "WebSocket sequence gaps detected",
                    gap_count=gap_count,
                    operation="ws_health",
                    stage="info",
                )

            # Update status reporter with WS health
            engine._status_reporter.update_ws_health(ws_health)

        except Exception:
            logger.exception("WS health watchdog error", operation="ws_health")

        await asyncio.sleep(interval)
