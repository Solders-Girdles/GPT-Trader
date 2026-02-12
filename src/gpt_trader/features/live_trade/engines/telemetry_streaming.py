from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import threading
import time
from collections.abc import Awaitable, Coroutine
from typing import TYPE_CHECKING, Any, cast

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover
    pass  # No type-only imports needed currently

logger = get_logger(__name__, component="telemetry_streaming")


STREAM_RETRY_BASE_SECONDS = 0.5
STREAM_RETRY_MULTIPLIER = 2.0
STREAM_RETRY_MAX_SECONDS = 60.0
STREAM_RETRY_MAX_ATTEMPTS = 5
WS_STREAM_RETRY_EVENT = "ws_stream_retry"
WS_STREAM_RETRY_EXHAUSTED_EVENT = "ws_stream_retry_exhausted"


@dataclass
class StreamingRetryState:
    """Track retry progress for telemetry streaming loops."""

    base_delay: float = STREAM_RETRY_BASE_SECONDS
    multiplier: float = STREAM_RETRY_MULTIPLIER
    max_delay: float = STREAM_RETRY_MAX_SECONDS
    max_attempts: int = STREAM_RETRY_MAX_ATTEMPTS
    attempts: int = 0

    def next_delay(self) -> float:
        delay = min(self.base_delay * (self.multiplier**self.attempts), self.max_delay)
        self.attempts += 1
        return delay

    def exhausted(self) -> bool:
        if self.max_attempts <= 0:
            return False
        return self.attempts >= self.max_attempts

    def reset(self) -> None:
        self.attempts = 0


def _get_retry_state(coordinator: Any) -> StreamingRetryState:
    state = getattr(coordinator, "_stream_retry_state", None)
    if not isinstance(state, StreamingRetryState):
        state = StreamingRetryState()
        setattr(coordinator, "_stream_retry_state", state)
    return state


def _consume_gap_count(coordinator: Any) -> int:
    gap_count = getattr(coordinator, "_stream_gap_count", 0)
    setattr(coordinator, "_stream_gap_count", 0)
    return gap_count


def _stop_signal_active(stop_signal: threading.Event | None) -> bool:
    if stop_signal is None:
        return False
    checker = getattr(stop_signal, "is_set", None)
    if callable(checker):
        try:
            result = checker()
            return bool(result) if isinstance(result, bool) else False
        except Exception as exc:
            logger.error(
                "Failed to check stop signal",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="run_stream_loop",
                stage="stop_signal_check",
            )
            return False
    if isinstance(stop_signal, bool):
        return bool(stop_signal)
    return False


def _supports_kwarg(fn: Any, keyword: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return keyword in params or any(param.kind == param.VAR_KEYWORD for param in params.values())


def _emit_metric(event_store: Any, bot_id: str, payload: dict[str, Any]) -> None:
    """Emit a metric event via the telemetry utility."""
    from gpt_trader.utilities.telemetry import emit_metric

    emit_metric(event_store, bot_id, payload)


def start_streaming_background(coordinator: Any) -> None:
    if not coordinator._should_enable_streaming():
        return
    coordinator._schedule_coroutine(coordinator._start_streaming())


def stop_streaming_background(coordinator: Any) -> None:
    coordinator._schedule_coroutine(coordinator._stop_streaming())


def _should_enable_streaming(coordinator: Any) -> bool:
    config = coordinator.context.config
    profile = getattr(config, "profile", None)
    if profile is not None and hasattr(profile, "value"):
        profile_name = profile.value
    else:
        profile_name = str(profile or "").lower()
    normalized = profile_name.lower()
    if normalized == "test":
        return False
    return bool(getattr(config, "perps_enable_streaming", False)) and normalized in {
        "canary",
        "prod",
    }


def restart_streaming_if_needed(
    coordinator: Any,
    diff: dict[str, Any],
) -> None:
    relevant = {"perps_enable_streaming", "perps_stream_level", "symbols"}
    if not relevant.intersection(diff.keys()):
        return

    should_stream = coordinator._should_enable_streaming()
    if "perps_enable_streaming" in diff:
        raw_value = str(diff["perps_enable_streaming"]).strip().lower()
        should_stream = raw_value in {"1", "true", "yes", "on"}
    if not should_stream and "symbols" in diff:
        should_stream = bool(coordinator.context.symbols)

    async def _apply_restart() -> None:
        try:
            maybe_stop = coordinator._stop_streaming()
            if inspect.isawaitable(maybe_stop):
                await asyncio.ensure_future(cast(Awaitable[Any], maybe_stop))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception(
                "Failed to stop streaming before restart",
                error=str(exc),
                operation="telemetry_stream",
                stage="restart_stop",
            )
        if should_stream:
            try:
                maybe_start = coordinator._start_streaming()
                if inspect.isawaitable(maybe_start):
                    await asyncio.ensure_future(cast(Awaitable[Any], maybe_start))
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception(
                    "Failed to restart streaming after config change",
                    error=str(exc),
                    operation="telemetry_stream",
                    stage="restart_start",
                )

    coro = _apply_restart()
    try:
        coordinator._schedule_coroutine(coro)
    except RuntimeError as exc:
        message = str(exc)
        if "asyncio.run()" in message and "running event loop" in message:
            coro.close()
            thread_error: Exception | None = None

            def _run_in_thread() -> None:
                nonlocal thread_error
                try:
                    asyncio.run(_apply_restart())
                except Exception as thread_exc:  # pragma: no cover - defensive handling
                    thread_error = thread_exc

            runner = threading.Thread(target=_run_in_thread, daemon=True)
            runner.start()
            runner.join()
            if thread_error:
                raise thread_error
        else:
            coro.close()
            raise


def _schedule_coroutine(
    coordinator: Any,
    coro: Coroutine[Any, Any, Any],
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError as exc:
        logger.error(
            "No running event loop for schedule_coroutine",
            error_type=type(exc).__name__,
            error_message=str(exc),
            operation="schedule_coroutine",
            stage="get_loop",
        )
        loop = None

    if loop and loop.is_running():
        loop.create_task(coro)
        return

    loop_task_handle = coordinator._loop_task_handle
    if loop_task_handle is not None:
        try:
            task_loop = loop_task_handle.get_loop()
            if task_loop.is_running():
                task_loop.call_soon_threadsafe(asyncio.create_task, coro)
                return
        except Exception as exc:
            logger.error(
                "Failed to schedule coroutine via loop_task_handle",
                error_type=type(exc).__name__,
                error_message=str(exc),
                operation="schedule_coroutine",
                stage="task_handle",
            )

    if loop is None:
        asyncio.run(coro)
    else:
        loop.run_until_complete(coro)


async def _start_streaming(
    coordinator: Any,
) -> asyncio.Task[Any] | None:
    symbols = list(coordinator.context.symbols)
    if not symbols:
        logger.debug(
            "No symbols configured; skipping streaming",
            operation="telemetry_stream",
            stage="skip",
        )
        return None

    configured_level = coordinator.context.config.perps_stream_level or 1
    try:
        level = max(int(configured_level), 1)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid streaming level; defaulting to 1",
            configured_level=configured_level,
            operation="telemetry_stream",
            stage="config",
        )
        level = 1

    coordinator._ws_stop = threading.Event()
    coordinator._pending_stream_config = (symbols, level)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug(
            "No running event loop; streaming will be deferred",
            operation="telemetry_stream",
            stage="deferred",
        )
        return None

    task = loop.create_task(
        coordinator._run_stream_loop_async(symbols, level, coordinator._ws_stop)
    )
    task.add_done_callback(coordinator._handle_stream_task_completion)
    coordinator._stream_task = task
    coordinator._loop_task_handle = task
    logger.info(
        "Started WS streaming task",
        symbols=symbols,
        stream_level=level,
        operation="telemetry_stream",
        stage="start",
    )
    return task


async def _stop_streaming(coordinator: Any) -> None:
    coordinator._pending_stream_config = None
    stop_signal = coordinator._ws_stop
    if stop_signal is not None:
        stop_signal.set()
        coordinator._ws_stop = None

    task = coordinator._stream_task
    if task and callable(getattr(task, "done", None)) and not task.done():
        cancel = getattr(task, "cancel", None)
        if callable(cancel):
            cancel()
        try:
            if inspect.isawaitable(task):
                await asyncio.ensure_future(cast(Awaitable[Any], task))
            else:
                result = getattr(task, "result", None)
                if callable(result):
                    try:
                        result()
                    except Exception as exc:
                        logger.error(
                            "Failed to get streaming task result",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            operation="stop_streaming",
                            stage="get_result",
                        )
        except asyncio.CancelledError:
            logger.info(
                "WS streaming task cancelled",
                operation="telemetry_stream",
                stage="cancel",
            )
    coordinator._stream_task = None
    coordinator._loop_task_handle = None
    logger.info(
        "Streaming halted",
        operation="telemetry_stream",
        stage="stop",
    )


def _handle_stream_task_completion(
    coordinator: Any,
    task: asyncio.Task[Any],
) -> None:
    coordinator._stream_task = None
    coordinator._ws_stop = None
    try:
        task.result()
    except asyncio.CancelledError:
        logger.info(
            "WS streaming task cancelled",
            operation="telemetry_stream",
            stage="cancel",
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception(
            "WS streaming task failed",
            error=str(exc),
            operation="telemetry_stream",
            stage="failed",
        )


async def _run_stream_loop_async(
    coordinator: Any,
    symbols: list[str],
    level: int,
    stop_signal: threading.Event | None,
) -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None,
            coordinator._run_stream_loop,
            symbols,
            level,
            stop_signal,
        )
    except asyncio.CancelledError:
        if stop_signal:
            stop_signal.set()
        raise


def _run_stream_loop(
    coordinator: Any,
    symbols: list[str],
    level: int,
    stop_signal: threading.Event | None,
) -> None:
    ctx = coordinator.context
    broker = ctx.broker
    if broker is None:
        logger.error(
            "Cannot start streaming: no broker available",
            operation="telemetry_stream",
            stage="run",
        )
        return

    user_handler = None
    handler_state = getattr(coordinator, "__dict__", {})
    if isinstance(handler_state, dict) and "_user_event_handler" in handler_state:
        user_handler = handler_state.get("_user_event_handler")
    include_user_events = user_handler is not None
    retry_state = _get_retry_state(coordinator)
    coordinator._stream_gap_count = 0

    try:
        while True:
            if _stop_signal_active(stop_signal):
                retry_state.reset()
                break
            try:
                stream: Any | None = None
                try:
                    if include_user_events and _supports_kwarg(broker.stream_orderbook, "include_user_events"):
                        stream = broker.stream_orderbook(
                            symbols,
                            level=level,
                            include_trades=True,
                            include_user_events=True,
                        )
                    else:
                        stream = broker.stream_orderbook(symbols, level=level, include_trades=True)
                except Exception as exc:  # pragma: no cover - dependent on broker impl
                    logger.warning(
                        f"Orderbook stream unavailable, falling back to trades ({exc})",
                        error=str(exc),
                        operation="telemetry_stream",
                        stage="orderbook",
                    )
                    try:
                        stream = broker.stream_trades(symbols)
                    except Exception as trade_exc:
                        logger.error(
                            f"Failed to start streaming trades ({trade_exc})",
                            error=str(trade_exc),
                            operation="telemetry_stream",
                            stage="trades",
                        )
                        raise RuntimeError("Failed to start telemetry stream") from trade_exc

                if include_user_events:
                    backfill = getattr(user_handler, "request_backfill", None)
                    if callable(backfill):
                        backfill(reason="startup")

                should_stop = False
                for msg in stream or []:
                    if _stop_signal_active(stop_signal):
                        should_stop = True
                        break
                    if not isinstance(msg, dict):
                        continue
                    ctx = coordinator.context

                    channel = msg.get("channel", "")

                    if msg.get("gap_detected"):
                        coordinator._stream_gap_count = (
                            getattr(coordinator, "_stream_gap_count", 0) + 1
                        )
                        gap_backfill = getattr(user_handler, "request_backfill", None)
                        if callable(gap_backfill):
                            gap_backfill(reason="sequence_gap")

                    if channel == "user":
                        handler = getattr(user_handler, "handle_user_message", None)
                        if callable(handler):
                            try:
                                handler(msg)
                            except Exception as exc:
                                logger.debug(
                                    "Failed to handle user event message",
                                    error=str(exc),
                                    operation="telemetry_stream",
                                    stage="user_event_handler",
                                )
                        continue

                    if channel == "l2_data":
                        _handle_orderbook_message(coordinator, ctx, msg)
                    elif channel == "market_trades":
                        _handle_trade_message(coordinator, ctx, msg)
                    else:
                        sym = str(msg.get("product_id") or msg.get("symbol") or "")
                        if not sym:
                            continue

                        mark = coordinator._extract_mark_from_message(msg)
                        if mark is None or mark <= 0:
                            continue

                        coordinator._update_mark_and_metrics(ctx, sym, mark)

                retry_state.reset()
                if should_stop:
                    break
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                gap_count = _consume_gap_count(coordinator)
                ctx = coordinator.context
                _emit_metric(
                    ctx.event_store,
                    ctx.bot_id,
                    {
                        "event_type": "ws_stream_error",
                        "message": str(exc),
                        "gap_count": gap_count,
                        "attempts": retry_state.attempts,
                    },
                )
                if _stop_signal_active(stop_signal):
                    break
                if retry_state.exhausted():
                    _emit_metric(
                        ctx.event_store,
                        ctx.bot_id,
                        {
                            "event_type": WS_STREAM_RETRY_EXHAUSTED_EVENT,
                            "error": str(exc),
                            "gap_count": gap_count,
                            "attempts": retry_state.attempts,
                        },
                    )
                    break
                delay = retry_state.next_delay()
                _emit_metric(
                    ctx.event_store,
                    ctx.bot_id,
                    {
                        "event_type": WS_STREAM_RETRY_EVENT,
                        "delay_seconds": delay,
                        "error": str(exc),
                        "gap_count": gap_count,
                        "attempts": retry_state.attempts,
                    },
                )
                time.sleep(delay)
                continue
    finally:
        ctx = coordinator.context
        gap_count = _consume_gap_count(coordinator)
        _emit_metric(
            ctx.event_store,
            ctx.bot_id,
            {"event_type": "ws_stream_exit", "gap_count": gap_count},
        )


def _handle_orderbook_message(coordinator: Any, ctx: Any, msg: dict) -> None:
    """Handle level2 orderbook update message."""
    from gpt_trader.features.brokerages.coinbase.ws_events import OrderbookUpdate
    from gpt_trader.features.live_trade.engines.telemetry_health import (
        update_orderbook_snapshot,
    )

    try:
        event = OrderbookUpdate.from_message(msg)
        if event.product_id:
            update_orderbook_snapshot(ctx, event)

            # Also extract mark from orderbook for price tracking
            if event.bids and event.asks:
                best_bid = event.bids[0][0] if event.bids else None
                best_ask = event.asks[0][0] if event.asks else None
                if best_bid and best_ask:
                    from decimal import Decimal

                    mark = (best_bid + best_ask) / Decimal("2")
                    if mark > 0:
                        coordinator._update_mark_and_metrics(ctx, event.product_id, mark)
    except Exception as exc:
        logger.debug(
            "Failed to handle orderbook message",
            error=str(exc),
            operation="telemetry_stream",
            stage="orderbook_handler",
        )


def _handle_trade_message(coordinator: Any, ctx: Any, msg: dict) -> None:
    """Handle market_trades message."""
    from gpt_trader.features.brokerages.coinbase.ws_events import TradeEvent
    from gpt_trader.features.live_trade.engines.telemetry_health import (
        update_trade_aggregator,
    )

    try:
        events = TradeEvent.from_message(msg)
        for event in events:
            if event.product_id:
                update_trade_aggregator(ctx, event)

                # Also use trade price for mark tracking
                if event.price and event.price > 0:
                    coordinator._update_mark_and_metrics(ctx, event.product_id, event.price)
    except Exception as exc:
        logger.debug(
            "Failed to handle trade message",
            error=str(exc),
            operation="telemetry_stream",
            stage="trade_handler",
        )


__all__ = [
    "start_streaming_background",
    "stop_streaming_background",
    "restart_streaming_if_needed",
    "_schedule_coroutine",
    "_start_streaming",
    "_stop_streaming",
    "_handle_stream_task_completion",
    "_run_stream_loop_async",
    "_run_stream_loop",
    "_handle_orderbook_message",
    "_handle_trade_message",
    "_should_enable_streaming",
]
