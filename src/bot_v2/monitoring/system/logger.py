"""
Production logger with JSON structured logging and correlation ID tracking.

Features:
- JSON structured logging
- Correlation ID tracking for request tracing
- <5ms overhead performance
- Specialized methods for different event types
- Emits JSON lines to rotating file handlers via `logging` (bot_v2.json)
"""

import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any

from bot_v2.monitoring.system.correlation_context import CorrelationContext
from bot_v2.monitoring.system.log_buffer import LogBuffer
from bot_v2.monitoring.system.log_emitter import LogEmitter
from bot_v2.monitoring.system.performance_tracker import PerformanceTracker


class LogLevel(Enum):
    """Log severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class ProductionLogger:
    """High-performance production logger with structured JSON output."""

    def __init__(self, service_name: str = "bot_v2", enable_console: bool = True) -> None:
        """
        Initialize production logger.

        Args:
            service_name: Name of the service for log tagging
            enable_console: Whether to print to console (disable for production)
        """
        self.service_name = service_name

        # Infrastructure components
        self._correlation = CorrelationContext()
        self._buffer = LogBuffer(max_size=1000)
        self._performance = PerformanceTracker()

        # Configure emitter
        min_level = os.getenv("PERPS_MIN_LOG_LEVEL", "info").lower()
        if os.getenv("PERPS_DEBUG") in ("1", "true", "yes", "on"):
            min_level = "debug"

        # Python logger for JSON lines; handlers configured in bot_v2.logging_setup
        py_logger = logging.getLogger(f"{service_name}.json")
        if not py_logger.handlers:
            py_logger = logging.getLogger("bot_v2.json")

        # Allow env var to override console logging to avoid noisy prod
        env_console = os.getenv("PERPS_JSON_CONSOLE")
        if env_console is not None:
            enable_console = env_console.strip().lower() in ("1", "true", "yes", "on")

        self._emitter = LogEmitter(
            service_name=service_name,
            enable_console=enable_console,
            min_level=min_level,
            py_logger=py_logger,
        )

    def set_correlation_id(self, correlation_id: str | None = None) -> None:
        """Set correlation ID for current thread."""
        self._correlation.set_correlation_id(correlation_id)

    def get_correlation_id(self) -> str:
        """Get current thread's correlation ID."""
        return self._correlation.get_correlation_id()

    def _create_log_entry(
        self, level: LogLevel, event_type: str, message: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Create structured log entry with minimal overhead."""
        start_time = time.perf_counter()

        # Build log entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level.value,
            "service": self.service_name,
            "correlation_id": self._correlation.get_correlation_id(),
            "event_type": event_type,
            "message": message,
        }

        # Add any additional fields
        if kwargs:
            entry.update(kwargs)

        # Track performance
        self._performance.record(time.perf_counter() - start_time)

        return entry

    def _emit_log(self, entry: dict[str, Any]) -> None:
        """Emit log entry (delegates to buffer and emitter)."""
        # Only buffer/emit if entry meets minimum level
        if self._emitter._should_emit(entry):  # noqa: SLF001
            self._buffer.append(entry)
            self._emitter.emit(entry)

    def log_event(
        self,
        level: LogLevel,
        event_type: str,
        message: str,
        component: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a general system event.

        Args:
            level: Log severity level
            event_type: Type of event (e.g., "system_start", "config_change")
            message: Human-readable message
            component: Component name (optional)
            **kwargs: Additional structured data
        """
        entry = self._create_log_entry(level, event_type, message, **kwargs)

        if component:
            entry["component"] = component

        self._emit_log(entry)

    def log_trade(
        self,
        action: str,  # "buy", "sell", "close"
        symbol: str,
        quantity: float,
        price: float,
        strategy: str,
        success: bool = True,
        execution_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log trading activity with specialized fields.

        Args:
            action: Trading action
            symbol: Trading symbol
            quantity: Trade quantity
            price: Execution price
            strategy: Strategy name
            success: Whether trade was successful
            execution_time_ms: Execution time in milliseconds
            **kwargs: Additional trade data
        """
        level = LogLevel.INFO if success else LogLevel.ERROR

        entry = self._create_log_entry(
            level=level,
            event_type="trade_execution",
            message=f"{action.upper()} {quantity} {symbol} @ {price}",
            trade_action=action,
            symbol=symbol,
            quantity=quantity,
            price=price,
            strategy=strategy,
            success=success,
            **kwargs,
        )

        if execution_time_ms is not None:
            entry["execution_time_ms"] = execution_time_ms

        self._emit_log(entry)

    def log_ml_prediction(
        self,
        model_name: str,
        prediction: Any,
        confidence: float | None = None,
        input_features: dict[str, Any] | None = None,
        inference_time_ms: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Log ML model predictions and inference.

        Args:
            model_name: Name of the ML model
            prediction: Model prediction result
            confidence: Prediction confidence score (0-1)
            input_features: Input features (summary)
            inference_time_ms: Inference time in milliseconds
            **kwargs: Additional ML data
        """
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="ml_prediction",
            message=f"Model {model_name} predicted: {prediction}",
            model_name=model_name,
            prediction=str(prediction),  # Convert to string for JSON safety
            **kwargs,
        )

        if confidence is not None:
            entry["confidence"] = confidence

        if input_features:
            # Summarize features to avoid large logs
            entry["feature_count"] = len(input_features)
            entry["sample_features"] = {k: v for k, v in list(input_features.items())[:5]}

        if inference_time_ms is not None:
            entry["inference_time_ms"] = inference_time_ms

        self._emit_log(entry)

    def log_performance(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Log performance metrics for operations.

        Args:
            operation: Operation name
            duration_ms: Operation duration in milliseconds
            success: Whether operation succeeded
            **kwargs: Additional performance data
        """
        level = LogLevel.INFO if success else LogLevel.WARNING

        entry = self._create_log_entry(
            level=level,
            event_type="performance_metric",
            message=f"{operation} completed in {duration_ms:.2f}ms",
            operation=operation,
            duration_ms=duration_ms,
            success=success,
            **kwargs,
        )

        self._emit_log(entry)

    def log_error(self, error: Exception, context: str | None = None, **kwargs: Any) -> None:
        """
        Log errors with full context.

        Args:
            error: Exception object
            context: Additional context about when error occurred
            **kwargs: Additional error data
        """
        entry = self._create_log_entry(
            level=LogLevel.ERROR,
            event_type="error",
            message=str(error),
            error_type=type(error).__name__,
            **kwargs,
        )

        if context:
            entry["error_context"] = context

        self._emit_log(entry)

    # --- Specialized audit and domain logs ---
    def log_auth_event(
        self,
        action: str,  # "jwt_generate", "jwt_refresh", "key_rotation", "auth_failure"
        provider: str,  # e.g., "coinbase_cdp"
        success: bool,
        error_code: str | None = None,
        **kwargs: Any,
    ) -> None:
        level = LogLevel.INFO if success else LogLevel.ERROR
        msg = f"auth {action} ({provider}) {'ok' if success else 'failed'}"
        entry = self._create_log_entry(
            level=level,
            event_type="auth_event",
            message=msg,
            action=action,
            provider=provider,
            success=success,
            error_code=error_code,
            **kwargs,
        )
        self._emit_log(entry)

    def log_pnl(
        self,
        symbol: str,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        fees: float | None = None,
        funding: float | None = None,
        position_size: float | None = None,
        transition: str | None = None,  # e.g., "unrealized->realized"
        **kwargs: Any,
    ) -> None:
        msg = f"PnL update {symbol}"
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="pnl_update",
            message=msg,
            symbol=symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            fees=fees,
            funding=funding,
            position_size=position_size,
            transition=transition,
            **kwargs,
        )
        self._emit_log(entry)

    def log_funding(
        self,
        symbol: str,
        funding_rate: float,
        payment: float,
        period_start: str | None = None,
        period_end: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="funding_applied",
            message=f"Funding applied {symbol} rate={funding_rate}",
            symbol=symbol,
            funding_rate=funding_rate,
            payment=payment,
            period_start=period_start,
            period_end=period_end,
            **kwargs,
        )
        self._emit_log(entry)

    def log_market_heartbeat(
        self,
        source: str,
        last_update_ts: str,
        latency_ms: float | None = None,
        staleness_ms: float | None = None,
        threshold_ms: int | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="market_heartbeat",
            message=f"market heartbeat {source}",
            source=source,
            last_update_ts=last_update_ts,
            latency_ms=latency_ms,
            staleness_ms=staleness_ms,
            staleness_threshold_ms=threshold_ms,
            **kwargs,
        )
        self._emit_log(entry)

    def log_order_submission(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="order_submission",
            message=f"submit {side} {quantity} {symbol} @{price if price is not None else 'mkt'}",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            **kwargs,
        )
        self._emit_log(entry)

    def log_order_status_change(
        self,
        order_id: str,
        client_order_id: str | None,
        from_status: str | None,
        to_status: str,
        exchange_error_code: str | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        level = (
            LogLevel.WARNING if to_status in ("REJECTED", "CANCELLED", "EXPIRED") else LogLevel.INFO
        )
        entry = self._create_log_entry(
            level=level,
            event_type="order_status_change",
            message=f"order {order_id} → {to_status}",
            order_id=order_id,
            client_order_id=client_order_id,
            from_status=from_status,
            to_status=to_status,
            exchange_error_code=exchange_error_code,
            reason=reason,
            **kwargs,
        )
        self._emit_log(entry)

    def log_position_change(
        self,
        symbol: str,
        side: str,
        size: float,
        avg_entry_price: float | None = None,
        realized_pnl: float | None = None,
        unrealized_pnl: float | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="position_change",
            message=f"position {symbol} {side} size={size}",
            symbol=symbol,
            side=side,
            size=size,
            avg_entry_price=avg_entry_price,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            **kwargs,
        )
        self._emit_log(entry)

    def log_balance_update(
        self,
        currency: str,
        available: float,
        total: float,
        change: float | None = None,
        reason: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="balance_update",
            message=f"balance {currency} total={total}",
            currency=currency,
            available=available,
            total=total,
            change=change,
            reason=reason,
            **kwargs,
        )
        self._emit_log(entry)

    def log_risk_breach(
        self,
        limit_type: str,
        limit_value: float,
        current_value: float,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.WARNING,
            event_type="risk_limit_breach",
            message=f"risk breach {limit_type}",
            limit_type=limit_type,
            limit_value=limit_value,
            current_value=current_value,
            exceeded_by=current_value - limit_value,
            **kwargs,
        )
        self._emit_log(entry)

    def log_order_round_trip(
        self,
        order_id: str,
        client_order_id: str | None,
        round_trip_ms: float,
        submitted_ts: str | None = None,
        filled_ts: str | None = None,
        **kwargs: Any,
    ) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="order_round_trip",
            message=f"order {order_id} rtt={round_trip_ms:.2f}ms",
            order_id=order_id,
            client_order_id=client_order_id,
            round_trip_ms=round_trip_ms,
            submitted_ts=submitted_ts,
            filled_ts=filled_ts,
            **kwargs,
        )
        self._emit_log(entry)

    def log_ws_latency(self, stream: str, latency_ms: float, **kwargs: Any) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="ws_latency",
            message=f"ws {stream} latency={latency_ms:.2f}ms",
            stream=stream,
            latency_ms=latency_ms,
            **kwargs,
        )
        self._emit_log(entry)

    def log_rest_response(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        **kwargs: Any,
    ) -> None:
        level = LogLevel.INFO if 200 <= status_code < 400 else LogLevel.WARNING
        entry = self._create_log_entry(
            level=level,
            event_type="rest_timing",
            message=f"{method.upper()} {endpoint} {status_code} in {duration_ms:.1f}ms",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            duration_ms=duration_ms,
            **kwargs,
        )
        self._emit_log(entry)

    def log_strategy_duration(self, strategy: str, duration_ms: float, **kwargs: Any) -> None:
        entry = self._create_log_entry(
            level=LogLevel.INFO,
            event_type="strategy_duration",
            message=f"{strategy} took {duration_ms:.1f}ms",
            strategy=strategy,
            duration_ms=duration_ms,
            **kwargs,
        )
        self._emit_log(entry)

    def get_recent_logs(self, count: int = 100) -> list[dict[str, Any]]:
        """Get recent log entries."""
        return self._buffer.get_recent(count)

    def get_performance_stats(self) -> dict[str, float | int]:
        """Get logger performance statistics."""
        return self._performance.get_stats()


# Global logger instance
_logger: ProductionLogger | None = None


def get_logger(service_name: str = "bot_v2", enable_console: bool = True) -> ProductionLogger:
    """
    Get or create global logger instance.

    Args:
        service_name: Service name for logging
        enable_console: Whether to enable console output

    Returns:
        ProductionLogger instance
    """
    global _logger

    if _logger is None:
        _logger = ProductionLogger(service_name, enable_console)

    return _logger


# Convenience functions for common logging patterns
def log_event(
    event_type: str, message: str, level: LogLevel = LogLevel.INFO, **kwargs: Any
) -> None:
    """Log a system event."""
    logger = get_logger()
    logger.log_event(level, event_type, message, **kwargs)


def log_trade(
    action: str, symbol: str, quantity: float, price: float, strategy: str, **kwargs: Any
) -> None:
    """Log a trade execution."""
    logger = get_logger()
    logger.log_trade(action, symbol, quantity, price, strategy, **kwargs)


def log_ml_prediction(model_name: str, prediction: Any, **kwargs: Any) -> None:
    """Log an ML prediction."""
    logger = get_logger()
    logger.log_ml_prediction(model_name, prediction, **kwargs)


def log_performance(operation: str, duration_ms: float, **kwargs: Any) -> None:
    """Log a performance metric."""
    logger = get_logger()
    logger.log_performance(operation, duration_ms, **kwargs)


def log_error(error: Exception, context: str | None = None, **kwargs: Any) -> None:
    """Log an error."""
    logger = get_logger()
    logger.log_error(error, context, **kwargs)


def set_correlation_id(correlation_id: str | None = None) -> None:
    """Set correlation ID for current thread."""
    logger = get_logger()
    logger.set_correlation_id(correlation_id)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    logger = get_logger()
    return logger.get_correlation_id()
