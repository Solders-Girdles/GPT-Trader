"""Error handling utilities for order placement."""

from __future__ import annotations

import os

from ...logging_utils import json_logger, logger


def handle_order_error(
    mixin: "OrderPlacementMixin",
    symbol: str,
    kwargs: dict[str, object],
    exc: Exception,
) -> None:
    """Handle validation or execution failures during order placement."""
    logger.warning(
        "Order validation/execution failed",
        error=str(exc),
        operation="execution_order",
        stage="validation",
    )
    json_logger.warning(
        "Order validation/execution failed",
        extra={
            "error": str(exc),
            "operation": "execution_order",
            "stage": "validation",
            "symbol": symbol,
        },
    )
    mixin._record_event(
        "validation_error",
        {
            "symbol": symbol,
            "error": str(exc),
        },
    )
    mixin._record_event(
        "risk_rejection",
        {
            "symbol": symbol,
            "error": str(exc),
            "order_id": str(
                kwargs.get("client_order_id")
                or kwargs.get("order_id")
                or os.getenv("INTEGRATION_TEST_ORDER_ID", "")
            ),
        },
    )
    broker_ref = mixin.context.broker
    override_exc: Exception | None = None
    if broker_ref is not None:
        if getattr(broker_ref, "connection_dropped", False):
            override_exc = ConnectionError("Broker connection dropped during order placement")
        elif getattr(broker_ref, "api_rate_limited", False):
            override_exc = RuntimeError("API rate limit exceeded")
        elif getattr(broker_ref, "maintenance_mode", False):
            override_exc = RuntimeError("Broker under maintenance")

    error_for_recording = override_exc or exc
    mixin._record_broker_error(error_for_recording, symbol=symbol)
    handle_risk_callback(mixin, error_for_recording, symbol)
    mixin._increment_order_stat("failed")


def handle_risk_callback(mixin: "OrderPlacementMixin", exc: Exception, symbol: str) -> None:
    """Invoke risk callbacks when available."""
    risk_manager = mixin.context.risk_manager
    if risk_manager is not None and hasattr(risk_manager, "handle_broker_error"):
        try:
            risk_manager.handle_broker_error(exc, {"symbol": symbol})
        except Exception:
            logger.debug(
                "Risk manager handle_broker_error raised",
                error=str(exc),
                operation="execution_order",
                stage="risk_manager_callback",
            )


__all__ = ["handle_order_error", "handle_risk_callback"]
