"""OpenTelemetry tracing integration.

Provides distributed tracing for the trading system. Tracing is opt-in and
requires the `observability` optional dependency group.

Usage:
    # Initialize at startup
    from gpt_trader.observability import init_tracing
    init_tracing(service_name="gpt-trader", endpoint="http://localhost:4317", enabled=True)

    # Create spans
    from gpt_trader.observability import trace_span
    with trace_span("cycle", {"cycle": 42, "symbol": "BTC-USD"}):
        # ... do work ...

When OTel is not installed or tracing is disabled, trace_span is a no-op.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from gpt_trader.logging.correlation import get_log_context
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tracing")

# Module-level state
_tracer: Any | None = None
_tracing_enabled: bool = False

# OTel SDK availability
_OTEL_AVAILABLE: bool = False
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:
    pass


def init_tracing(
    service_name: str = "gpt-trader",
    endpoint: str | None = None,
    enabled: bool = True,
) -> bool:
    """Initialize OpenTelemetry tracing.

    This function is idempotent - calling it multiple times with different
    parameters will reconfigure tracing.

    Args:
        service_name: Name of the service for span attribution.
        endpoint: OTLP gRPC endpoint (e.g., "http://localhost:4317").
            If None, spans are collected but not exported.
        enabled: Whether tracing is enabled. If False, trace_span is a no-op.

    Returns:
        True if tracing was successfully initialized, False otherwise.
    """
    global _tracer, _tracing_enabled

    if not enabled:
        _tracing_enabled = False
        _tracer = None
        logger.info(
            "Tracing disabled",
            operation="init_tracing",
            enabled=False,
        )
        return False

    if not _OTEL_AVAILABLE:
        logger.warning(
            "OpenTelemetry not installed. Install with: pip install gpt-trader[observability]",
            operation="init_tracing",
        )
        _tracing_enabled = False
        _tracer = None
        return False

    try:
        # Create resource with service name
        resource = Resource.create({"service.name": service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add span processor with exporter if endpoint is provided
        if endpoint:
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            logger.info(
                "Tracing initialized with OTLP exporter",
                operation="init_tracing",
                service_name=service_name,
                endpoint=endpoint,
            )
        else:
            logger.info(
                "Tracing initialized without exporter (spans collected locally only)",
                operation="init_tracing",
                service_name=service_name,
            )

        # Set the global tracer provider
        otel_trace.set_tracer_provider(provider)

        # Get tracer for this module
        _tracer = otel_trace.get_tracer(__name__)
        _tracing_enabled = True

        return True

    except Exception as exc:
        logger.warning(
            "Failed to initialize tracing",
            operation="init_tracing",
            error=str(exc),
        )
        _tracing_enabled = False
        _tracer = None
        return False


def get_tracer() -> Any | None:
    """Get the current tracer instance.

    Returns:
        The OTel tracer if tracing is enabled and initialized, None otherwise.
    """
    return _tracer if _tracing_enabled else None


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled.

    Returns:
        True if tracing is enabled and the tracer is available.
    """
    return _tracing_enabled and _tracer is not None


@contextmanager
def trace_span(
    name: str,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any | None]:
    """Create a traced span with automatic context propagation.

    Automatically attaches correlation context (correlation_id, cycle, symbol,
    order_id) from the logging context to span attributes.

    Args:
        name: Name of the span (e.g., "cycle", "order_submit", "http_request").
        attributes: Additional attributes to attach to the span.

    Yields:
        The OTel span if tracing is enabled, None otherwise.
        The span can be used to add events or set status:

            with trace_span("operation") as span:
                if span:
                    span.add_event("checkpoint")
                    span.set_attribute("result", "success")
    """
    if not _tracing_enabled or _tracer is None:
        yield None
        return

    # Merge provided attributes with correlation context
    span_attrs: dict[str, Any] = {}

    # Pull correlation context
    log_context = get_log_context()
    for key, value in log_context.items():
        # OTel attributes must be strings, numbers, or bools
        if isinstance(value, (str, int, float, bool)):
            span_attrs[key] = value
        elif value is not None:
            span_attrs[key] = str(value)

    # Overlay provided attributes
    if attributes:
        for key, value in attributes.items():
            if isinstance(value, (str, int, float, bool)):
                span_attrs[key] = value
            elif value is not None:
                span_attrs[key] = str(value)

    with _tracer.start_as_current_span(name, attributes=span_attrs) as span:
        yield span


def shutdown_tracing() -> None:
    """Shutdown tracing and flush any pending spans.

    Call this during application shutdown to ensure all spans are exported.
    """
    global _tracer, _tracing_enabled

    if not _OTEL_AVAILABLE or not _tracing_enabled:
        return

    try:
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
            logger.info("Tracing shutdown complete", operation="shutdown_tracing")
    except Exception as exc:
        logger.warning(
            "Error during tracing shutdown",
            operation="shutdown_tracing",
            error=str(exc),
        )
    finally:
        _tracer = None
        _tracing_enabled = False


__all__ = [
    "init_tracing",
    "trace_span",
    "get_tracer",
    "is_tracing_enabled",
    "shutdown_tracing",
]
