"""Async workflow helpers for order placement."""

from __future__ import annotations

from typing import Any

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError
from bot_v2.logging import log_execution_error, order_context
from bot_v2.utilities.async_utils import run_in_thread

from ...logging_utils import json_logger, logger
from .errors import handle_order_error, handle_risk_callback
from .finalization import finalize_successful_order
from .resolution import resolve_order_from_result


async def place_order(
    mixin: "OrderPlacementMixin",
    exec_engine: Any,
    **kwargs: Any,
) -> Any:
    """Entry point for placing orders via the mixin."""
    symbol = kwargs.get("symbol", "unknown")
    with order_context("pending", symbol):
        lock = mixin.ensure_order_lock()
        try:
            mixin._last_exec_engine = exec_engine
            async with lock:
                return await place_order_inner(mixin, exec_engine, **kwargs)
        except (ValidationError, RiskValidationError, ExecutionError) as exc:
            handle_order_error(mixin, symbol, kwargs, exc)
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to place order",
                error=str(exc),
                exc_info=True,
                operation="execution_order",
                stage="submit_exception",
            )
            log_execution_error(error=exc, operation="place_order", symbol=symbol)
            mixin._record_broker_error(exc, symbol=symbol)
            handle_risk_callback(mixin, exc, symbol)
            mixin._increment_order_stat("failed")
            return None


async def place_order_inner(
    mixin: "OrderPlacementMixin",
    exec_engine: Any,
    **kwargs: Any,
) -> Any:
    """Perform the threaded broker call and resolve the result."""
    symbol = kwargs.get("symbol", "unknown")
    mixin._increment_order_stat("attempted")
    broker = mixin.context.broker
    runtime_state_obj = mixin.context.runtime_state

    if runtime_state_obj is None:
        logger.debug(
            "Runtime state missing; cannot record order",
            operation="execution_order",
            stage="runtime_state",
        )
        json_logger.debug(
            "Runtime state missing; cannot record order",
            extra={
                "operation": "execution_order",
                "stage": "runtime_state",
                "symbol": symbol,
            },
        )
        return None

    def _place() -> Any:
        return exec_engine.place_order(**kwargs)

    result = await run_in_thread(_place)

    order = await resolve_order_from_result(mixin, exec_engine, broker, result)
    if order:
        return await finalize_successful_order(mixin, order, kwargs)

    json_logger.warning(
        "Order attempt failed (no order returned)",
        extra={"operation": "execution_order", "stage": "record", "symbol": symbol},
    )
    failure_exc = RuntimeError("Invalid broker response: no order returned")
    mixin._record_event(
        "validation_error",
        {
            "symbol": symbol,
            "reason": "no_order_returned",
        },
    )
    mixin._record_broker_error(failure_exc, symbol=symbol)
    raise failure_exc


__all__ = ["place_order", "place_order_inner"]
