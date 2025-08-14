"""
Enhanced Structured Logging System with Correlation IDs and Distributed Tracing
Phase 3, Week 7: Operational Excellence
Tasks: OPS-001 to OPS-008

This module provides:
- JSON-formatted logs with consistent schema
- Correlation ID generation and propagation
- Distributed tracing across components
- Parent-child span relationships
- Automatic timing and latency tracking
- Integration with existing ML components
- OpenTelemetry compatibility
- High-performance logging (>10,000 logs/sec)
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import sys
import time
import traceback
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar

# Import OpenTelemetry if available
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

F = TypeVar("F", bound=Callable[..., Any])

# Context variables for correlation tracking
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")
trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
span_id: contextvars.ContextVar[str] = contextvars.ContextVar("span_id", default="")
parent_span_id: contextvars.ContextVar[str] = contextvars.ContextVar("parent_span_id", default="")


class LogLevel(str, Enum):
    """Enhanced log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # Special level for audit events
    METRIC = "METRIC"  # Special level for metrics


class LogFormat(str, Enum):
    """Log output formats."""

    JSON = "json"
    TEXT = "text"
    COLORED = "colored"
    OPENTELEMETRY = "opentelemetry"


class SpanType(str, Enum):
    """Types of spans for tracing."""

    HTTP_REQUEST = "http.request"
    DATABASE_QUERY = "db.query"
    ML_PREDICTION = "ml.prediction"
    ML_TRAINING = "ml.training"
    BACKTEST = "backtest"
    TRADE_EXECUTION = "trade.execution"
    RISK_CALCULATION = "risk.calculation"
    DATA_FETCH = "data.fetch"
    BUSINESS_LOGIC = "business.logic"
    SYSTEM_OPERATION = "system.operation"


@dataclass
class SpanContext:
    """Context information for a span."""

    span_id: str
    trace_id: str
    parent_span_id: str | None = None
    operation_name: str = ""
    span_type: SpanType = SpanType.SYSTEM_OPERATION
    start_time: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "started"
    error: str | None = None


@dataclass
class LogSchema:
    """Consistent log schema for all messages."""

    timestamp: str
    level: str
    logger: str
    message: str
    correlation_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation: str | None
    component: str
    service: str = "gpt-trader"
    version: str = "1.0.0"

    # Performance metrics
    duration_ms: float | None = None
    memory_mb: float | None = None
    cpu_percent: float | None = None

    # Context information
    module: str | None = None
    function: str | None = None
    line: int | None = None
    thread_id: int | None = None
    process_id: int | None = None

    # Business context
    symbol: str | None = None
    strategy: str | None = None
    model_id: str | None = None
    trade_id: str | None = None

    # Error information
    error_type: str | None = None
    error_message: str | None = None
    stack_trace: list[str] | None = None

    # Custom attributes
    attributes: dict[str, Any] = field(default_factory=dict)

    # Tags for aggregation
    tags: dict[str, str] = field(default_factory=dict)


class CorrelationIDGenerator:
    """Generates unique correlation IDs."""

    @staticmethod
    def generate() -> str:
        """Generate a new correlation ID."""
        return f"corr-{uuid.uuid4().hex[:16]}"

    @staticmethod
    def generate_trace_id() -> str:
        """Generate a new trace ID."""
        return f"trace-{uuid.uuid4().hex[:16]}"

    @staticmethod
    def generate_span_id() -> str:
        """Generate a new span ID."""
        return f"span-{uuid.uuid4().hex[:12]}"


class SpanManager:
    """Manages distributed tracing spans."""

    def __init__(self):
        self._spans: dict[str, SpanContext] = {}
        self._lock = Lock()

    def create_span(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.SYSTEM_OPERATION,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> SpanContext:
        """Create a new span."""
        span_id = CorrelationIDGenerator.generate_span_id()
        current_trace_id = trace_id.get()

        # If no trace ID exists, create one
        if not current_trace_id:
            current_trace_id = CorrelationIDGenerator.generate_trace_id()
            trace_id.set(current_trace_id)

        # Use current span as parent if not specified
        if parent_span_id is None:
            parent_span_id = span_id.get()

        span_context = SpanContext(
            span_id=span_id,
            trace_id=current_trace_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            span_type=span_type,
            attributes=attributes or {},
        )

        with self._lock:
            self._spans[span_id] = span_context

        return span_context

    def finish_span(self, span_id: str, status: str = "completed", error: str | None = None):
        """Finish a span."""
        with self._lock:
            if span_id in self._spans:
                span = self._spans[span_id]
                span.status = status
                span.error = error
                duration = time.time() - span.start_time
                span.attributes["duration_seconds"] = duration

    def get_span(self, span_id: str) -> SpanContext | None:
        """Get a span by ID."""
        with self._lock:
            return self._spans.get(span_id)

    def get_active_spans(self) -> list[SpanContext]:
        """Get all active spans."""
        with self._lock:
            return [span for span in self._spans.values() if span.status == "started"]


class HighPerformanceFormatter(logging.Formatter):
    """High-performance formatter for structured logging."""

    def __init__(
        self,
        format_type: LogFormat = LogFormat.JSON,
        include_performance: bool = True,
        include_context: bool = True,
        service_name: str = "gpt-trader",
        version: str = "1.0.0",
    ):
        super().__init__()
        self.format_type = format_type
        self.include_performance = include_performance
        self.include_context = include_context
        self.service_name = service_name
        self.version = version

        # Pre-compile color codes
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
            "AUDIT": "\033[34m",  # Blue
            "METRIC": "\033[37m",  # White
            "RESET": "\033[0m",  # Reset
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with minimal overhead."""
        try:
            if self.format_type == LogFormat.JSON:
                return self._format_json_fast(record)
            elif self.format_type == LogFormat.COLORED:
                return self._format_colored_fast(record)
            else:
                return self._format_text_fast(record)
        except Exception as e:
            # Fallback to basic formatting if something goes wrong
            return f"[LOG_FORMAT_ERROR] {record.getMessage()} | Error: {e}"

    def _format_json_fast(self, record: logging.LogRecord) -> str:
        """Fast JSON formatting with minimal allocations."""
        # Get context variables
        corr_id = correlation_id.get()
        tr_id = trace_id.get()
        sp_id = span_id.get()
        parent_sp_id = parent_span_id.get()

        # Build log data with minimal dict operations
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": corr_id,
            "trace_id": tr_id,
            "span_id": sp_id,
            "parent_span_id": parent_sp_id,
            "service": self.service_name,
            "version": self.version,
            "component": getattr(record, "component", record.name.split(".")[-1]),
        }

        # Add performance metrics if enabled
        if self.include_performance:
            log_data.update(
                {
                    "duration_ms": getattr(record, "duration_ms", None),
                    "memory_mb": getattr(record, "memory_mb", None),
                    "cpu_percent": getattr(record, "cpu_percent", None),
                }
            )

        # Add context information if enabled
        if self.include_context:
            log_data.update(
                {
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "thread_id": record.thread,
                    "process_id": record.process,
                }
            )

        # Add business context
        for attr in ["symbol", "strategy", "model_id", "trade_id", "operation"]:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)

        # Add exception information
        if record.exc_info:
            log_data.update(
                {
                    "error_type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                    "error_message": str(record.exc_info[1]) if record.exc_info[1] else None,
                    "stack_trace": traceback.format_exception(*record.exc_info),
                }
            )

        # Add custom attributes
        if hasattr(record, "attributes"):
            log_data["attributes"] = record.attributes

        # Add tags
        if hasattr(record, "tags"):
            log_data["tags"] = record.tags

        return json.dumps(log_data, separators=(",", ":"), default=str)

    def _format_colored_fast(self, record: logging.LogRecord) -> str:
        """Fast colored formatting for terminal output."""
        level_color = self.colors.get(record.levelname, self.colors["RESET"])
        reset = self.colors["RESET"]

        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        corr_id = correlation_id.get()

        return (
            f"{level_color}{record.levelname:<8}{reset} "
            f"{timestamp} "
            f"[{corr_id[:8] if corr_id else 'no-corr'}] "
            f"{record.name:<20} "
            f"| {record.getMessage()}"
        )

    def _format_text_fast(self, record: logging.LogRecord) -> str:
        """Fast text formatting."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        corr_id = correlation_id.get()

        return (
            f"{timestamp} | {record.levelname:<8} | "
            f"[{corr_id[:8] if corr_id else 'no-corr'}] | "
            f"{record.name} | {record.getMessage()}"
        )


class DistributedTracer:
    """Distributed tracing system compatible with OpenTelemetry."""

    def __init__(self, service_name: str = "gpt-trader"):
        self.service_name = service_name
        self.span_manager = SpanManager()

        # Initialize OpenTelemetry if available
        if OPENTELEMETRY_AVAILABLE:
            self._init_opentelemetry()

    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing."""
        trace.set_tracer_provider(TracerProvider())
        tracer_provider = trace.get_tracer_provider()

        # Add console exporter for development
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        self.tracer = trace.get_tracer(self.service_name)

    @contextmanager
    def start_span(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.SYSTEM_OPERATION,
        attributes: dict[str, Any] | None = None,
    ):
        """Start a new span with automatic cleanup."""
        # Create span
        span_context = self.span_manager.create_span(
            operation_name=operation_name, span_type=span_type, attributes=attributes
        )

        # Set context variables
        corr_token = None
        trace_token = None
        span_token = None
        parent_token = None

        try:
            # Generate correlation ID if not present
            current_corr_id = correlation_id.get()
            if not current_corr_id:
                current_corr_id = CorrelationIDGenerator.generate()
                corr_token = correlation_id.set(current_corr_id)

            # Set tracing context
            trace_token = trace_id.set(span_context.trace_id)
            parent_token = parent_span_id.set(span_context.parent_span_id or "")
            span_token = span_id.set(span_context.span_id)

            # Create OpenTelemetry span if available
            otel_span = None
            if OPENTELEMETRY_AVAILABLE and hasattr(self, "tracer"):
                otel_span = self.tracer.start_span(operation_name)
                if attributes:
                    for key, value in attributes.items():
                        otel_span.set_attribute(key, str(value))

            yield span_context

            # Mark span as completed
            self.span_manager.finish_span(span_context.span_id, "completed")

            if otel_span:
                otel_span.set_status(Status(StatusCode.OK))

        except Exception as e:
            # Mark span as failed
            self.span_manager.finish_span(span_context.span_id, "failed", str(e))

            if "otel_span" in locals() and otel_span:
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
                otel_span.record_exception(e)

            raise

        finally:
            # Clean up context
            if corr_token:
                correlation_id.reset(corr_token)
            if trace_token:
                trace_id.reset(trace_token)
            if span_token:
                span_id.reset(span_token)
            if parent_token:
                parent_span_id.reset(parent_token)

            if "otel_span" in locals() and otel_span:
                otel_span.end()


class EnhancedStructuredLogger:
    """
    Enhanced structured logger with correlation IDs and distributed tracing.

    Features:
    - High-performance JSON logging (>10,000 logs/sec)
    - Automatic correlation ID generation and propagation
    - Distributed tracing with span relationships
    - Performance metrics tracking
    - Business context enrichment
    - OpenTelemetry compatibility
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        format_type: LogFormat = LogFormat.JSON,
        log_file: Path | None = None,
        max_size_mb: int = 100,
        backup_count: int = 10,
        service_name: str = "gpt-trader",
        version: str = "1.0.0",
        enable_tracing: bool = True,
        performance_threshold_ms: float = 1000.0,
    ):
        """Initialize enhanced structured logger."""
        self.name = name
        self.service_name = service_name
        self.version = version
        self.performance_threshold_ms = performance_threshold_ms

        # Create base logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.propagate = False

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create high-performance formatter
        self.formatter = HighPerformanceFormatter(
            format_type=format_type, service_name=service_name, version=version
        )

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                str(log_file), maxBytes=max_size_mb * 1024 * 1024, backupCount=backup_count
            )
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        # Initialize distributed tracer
        self.tracer = DistributedTracer(service_name) if enable_tracing else None

        # Performance tracking
        self._operation_timers: dict[str, float] = {}

    def _create_log_record(
        self,
        level: str,
        message: str,
        operation: str | None = None,
        duration_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        **kwargs,
    ) -> logging.LogRecord:
        """Create an enhanced log record."""
        record = self.logger.makeRecord(
            name=self.name,
            level=getattr(logging, level.upper()),
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None,
        )

        # Add enhanced attributes
        record.component = self.name.split(".")[-1]
        record.operation = operation
        record.duration_ms = duration_ms
        record.attributes = attributes or {}
        record.tags = tags or {}

        # Add business context from kwargs
        for key in ["symbol", "strategy", "model_id", "trade_id"]:
            if key in kwargs:
                setattr(record, key, kwargs[key])

        return record

    def start_operation(self, operation_name: str) -> str:
        """Start timing an operation."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        self._operation_timers[operation_id] = time.time()
        return operation_id

    def end_operation(
        self,
        operation_id: str,
        operation_name: str | None = None,
        success: bool = True,
        **kwargs,
    ) -> float:
        """End timing an operation and log the result."""
        if operation_id not in self._operation_timers:
            return 0.0

        duration_seconds = time.time() - self._operation_timers.pop(operation_id)
        duration_ms = duration_seconds * 1000

        # Determine log level based on duration and success
        if not success:
            level = "ERROR"
        elif duration_ms > self.performance_threshold_ms:
            level = "WARNING"
        else:
            level = "INFO"

        # Extract operation name from ID if not provided
        if operation_name is None:
            operation_name = operation_id.split("_")[0]

        self._log(
            level=level,
            message=f"Operation {operation_name} completed in {duration_ms:.2f}ms",
            operation=operation_name,
            duration_ms=duration_ms,
            attributes={
                "operation_id": operation_id,
                "success": success,
                "duration_seconds": duration_seconds,
            },
            **kwargs,
        )

        return duration_ms

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        record = self._create_log_record(level, message, **kwargs)
        self.logger.handle(record)

    # Standard logging methods
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)

    def audit(self, message: str, **kwargs):
        """Log audit event."""
        self._log("AUDIT", message, **kwargs)

    def metric(self, message: str, **kwargs):
        """Log metric event."""
        self._log("METRIC", message, **kwargs)

    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        import sys

        exc_info = sys.exc_info()
        record = self._create_log_record("ERROR", message, **kwargs)
        record.exc_info = exc_info
        self.logger.handle(record)

    # Tracing methods
    def start_span(
        self,
        operation_name: str,
        span_type: SpanType = SpanType.SYSTEM_OPERATION,
        attributes: dict[str, Any] | None = None,
    ):
        """Start a distributed tracing span."""
        if self.tracer:
            return self.tracer.start_span(operation_name, span_type, attributes)
        else:
            # Fallback context manager
            @contextmanager
            def dummy_span():
                yield None

            return dummy_span()

    # Correlation ID methods
    def new_correlation_id(self) -> str:
        """Generate and set a new correlation ID."""
        corr_id = CorrelationIDGenerator.generate()
        correlation_id.set(corr_id)
        return corr_id

    def get_correlation_id(self) -> str:
        """Get current correlation ID."""
        return correlation_id.get()

    def set_correlation_id(self, corr_id: str):
        """Set correlation ID."""
        correlation_id.set(corr_id)

    @contextmanager
    def correlation_context(self, corr_id: str | None = None):
        """Context manager for correlation ID scope."""
        if corr_id is None:
            corr_id = CorrelationIDGenerator.generate()

        token = correlation_id.set(corr_id)
        try:
            yield corr_id
        finally:
            correlation_id.reset(token)


# Decorators for automatic tracing and logging
def traced_operation(
    operation_name: str | None = None,
    span_type: SpanType = SpanType.SYSTEM_OPERATION,
    log_args: bool = False,
    log_result: bool = False,
    performance_threshold_ms: float = 1000.0,
):
    """Decorator for automatic operation tracing and logging."""

    def decorator(func: F) -> F:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            with logger.start_span(op_name, span_type) as span:
                operation_id = logger.start_operation(op_name)

                try:
                    # Log function entry
                    log_kwargs = {}
                    if log_args:
                        log_kwargs["attributes"] = {
                            "args": str(args)[:200],
                            "kwargs": str(kwargs)[:200],
                        }

                    logger.debug(f"Starting {op_name}", operation=op_name, **log_kwargs)

                    # Execute function
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    # Log function exit
                    log_kwargs = {"duration_ms": duration_ms}
                    if log_result:
                        log_kwargs["attributes"] = {"result": str(result)[:200]}

                    if duration_ms > performance_threshold_ms:
                        logger.warning(
                            f"Slow operation {op_name} took {duration_ms:.2f}ms",
                            operation=op_name,
                            **log_kwargs,
                        )
                    else:
                        logger.debug(
                            f"Completed {op_name} in {duration_ms:.2f}ms",
                            operation=op_name,
                            **log_kwargs,
                        )

                    logger.end_operation(operation_id, op_name, success=True)
                    return result

                except Exception as e:
                    logger.end_operation(operation_id, op_name, success=False)
                    logger.exception(
                        f"Error in {op_name}: {e}",
                        operation=op_name,
                        attributes={"error_type": type(e).__name__},
                    )
                    raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)

            with logger.start_span(op_name, span_type) as span:
                operation_id = logger.start_operation(op_name)

                try:
                    # Log function entry
                    log_kwargs = {}
                    if log_args:
                        log_kwargs["attributes"] = {
                            "args": str(args)[:200],
                            "kwargs": str(kwargs)[:200],
                        }

                    logger.debug(f"Starting {op_name}", operation=op_name, **log_kwargs)

                    # Execute function
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    duration_ms = (time.time() - start_time) * 1000

                    # Log function exit
                    log_kwargs = {"duration_ms": duration_ms}
                    if log_result:
                        log_kwargs["attributes"] = {"result": str(result)[:200]}

                    if duration_ms > performance_threshold_ms:
                        logger.warning(
                            f"Slow operation {op_name} took {duration_ms:.2f}ms",
                            operation=op_name,
                            **log_kwargs,
                        )
                    else:
                        logger.debug(
                            f"Completed {op_name} in {duration_ms:.2f}ms",
                            operation=op_name,
                            **log_kwargs,
                        )

                    logger.end_operation(operation_id, op_name, success=True)
                    return result

                except Exception as e:
                    logger.end_operation(operation_id, op_name, success=False)
                    logger.exception(
                        f"Error in {op_name}: {e}",
                        operation=op_name,
                        attributes={"error_type": type(e).__name__},
                    )
                    raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Global logger registry
_logger_registry: dict[str, EnhancedStructuredLogger] = {}
_registry_lock = Lock()


def get_logger(
    name: str, level: str = "INFO", format_type: LogFormat = LogFormat.JSON, **kwargs
) -> EnhancedStructuredLogger:
    """Get or create a structured logger instance."""
    with _registry_lock:
        if name not in _logger_registry:
            _logger_registry[name] = EnhancedStructuredLogger(
                name=name, level=level, format_type=format_type, **kwargs
            )
        return _logger_registry[name]


def configure_logging(
    level: str = "INFO",
    format_type: LogFormat = LogFormat.JSON,
    log_file: Path | None = None,
    service_name: str = "gpt-trader",
    enable_tracing: bool = True,
):
    """Configure global logging settings."""
    global _logger_registry

    # Clear existing loggers
    with _registry_lock:
        _logger_registry.clear()

    # Set default configuration
    import logging

    logging.basicConfig(level=getattr(logging, level.upper()))

    # Create root logger
    get_logger(
        "gpt-trader",
        level=level,
        format_type=format_type,
        log_file=log_file,
        service_name=service_name,
        enable_tracing=enable_tracing,
    )


# Performance monitoring utilities
class LogPerformanceMonitor:
    """Monitor logging performance to ensure targets are met."""

    def __init__(self):
        self.start_time = time.time()
        self.log_count = 0
        self.total_duration = 0.0

    def record_log(self, duration: float):
        """Record a log operation."""
        self.log_count += 1
        self.total_duration += duration

    def get_stats(self) -> dict[str, float]:
        """Get performance statistics."""
        elapsed = time.time() - self.start_time
        return {
            "logs_per_second": self.log_count / elapsed if elapsed > 0 else 0,
            "average_latency_ms": (
                (self.total_duration / self.log_count * 1000) if self.log_count > 0 else 0
            ),
            "total_logs": self.log_count,
            "elapsed_seconds": elapsed,
        }


# Export public interface
__all__ = [
    "EnhancedStructuredLogger",
    "LogLevel",
    "LogFormat",
    "SpanType",
    "DistributedTracer",
    "traced_operation",
    "get_logger",
    "configure_logging",
    "CorrelationIDGenerator",
    "LogPerformanceMonitor",
]
