from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Coroutine
from typing import TYPE_CHECKING, Any, cast

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:  # pragma: no cover
    pass  # No type-only imports needed currently

logger = get_logger(__name__, component="telemetry_streaming")


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

    stream: Any | None = None
    try:
        stream = broker.stream_orderbook(symbols, level=level)
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
            stream = None

    try:
        for msg in stream or []:
            should_stop = False
            if stop_signal is not None:
                checker = getattr(stop_signal, "is_set", None)
                if callable(checker):
                    try:
                        result = checker()
                        should_stop = bool(result) if isinstance(result, bool) else False
                    except Exception as exc:
                        logger.error(
                            "Failed to check stop signal",
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            operation="run_stream_loop",
                            stage="stop_signal_check",
                        )
                        should_stop = False
                elif isinstance(stop_signal, bool):
                    should_stop = stop_signal
            if should_stop:
                break
            if not isinstance(msg, dict):
                continue
            ctx = coordinator.context
            sym = str(msg.get("product_id") or msg.get("symbol") or "")
            if not sym:
                continue

            mark = coordinator._extract_mark_from_message(msg)
            if mark is None or mark <= 0:
                continue

            coordinator._update_mark_and_metrics(ctx, sym, mark)
    except Exception as exc:  # pragma: no cover - defensive logging
        ctx = coordinator.context
        _emit_metric(
            ctx.event_store,
            ctx.bot_id,
            {"event_type": "ws_stream_error", "message": str(exc)},
        )
    finally:
        ctx = coordinator.context
        _emit_metric(
            ctx.event_store,
            ctx.bot_id,
            {"event_type": "ws_stream_exit"},
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
    "_should_enable_streaming",
]
