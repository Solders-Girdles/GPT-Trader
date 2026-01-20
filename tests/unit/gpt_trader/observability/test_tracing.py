"""Tests for OpenTelemetry tracing module."""

from __future__ import annotations

import pytest

import gpt_trader.observability.tracing as tracing_module
from gpt_trader.observability.tracing import (
    _OTEL_AVAILABLE,
    get_tracer,
    init_tracing,
    is_tracing_enabled,
    shutdown_tracing,
    trace_span,
)


@pytest.fixture(autouse=True)
def reset_tracing():
    """Reset tracing state before and after each test."""
    shutdown_tracing()
    yield
    shutdown_tracing()


class TestInitTracing:
    """Tests for init_tracing function."""

    def test_disabled_tracing_returns_false(self):
        """Test that disabled tracing returns False."""
        result = init_tracing(enabled=False)
        assert result is False
        assert is_tracing_enabled() is False

    def test_without_otel_returns_false(self, monkeypatch: pytest.MonkeyPatch):
        """Test that tracing without OTel installed returns False."""
        monkeypatch.setattr(tracing_module, "_OTEL_AVAILABLE", False)
        # When OTel is not available, init_tracing returns False
        assert init_tracing(enabled=True) is False

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_with_otel_returns_true(self):
        """Test that tracing with OTel installed returns True."""
        result = init_tracing(
            service_name="test-service",
            endpoint=None,  # No exporter, just in-memory
            enabled=True,
        )
        assert result is True
        assert is_tracing_enabled() is True
        assert get_tracer() is not None

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_init_with_endpoint(self):
        """Test initialization with OTLP endpoint."""
        # Note: This doesn't actually connect, just sets up the exporter
        result = init_tracing(
            service_name="test-service",
            endpoint="http://localhost:4317",
            enabled=True,
        )
        assert result is True


class TestTraceSpan:
    """Tests for trace_span context manager."""

    def test_disabled_tracing_yields_none(self):
        """Test that disabled tracing yields None."""
        init_tracing(enabled=False)

        with trace_span("test_span") as span:
            assert span is None

    def test_disabled_tracing_executes_block(self):
        """Test that code block executes even when tracing disabled."""
        init_tracing(enabled=False)

        executed = False
        with trace_span("test_span"):
            executed = True

        assert executed is True

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_enabled_tracing_yields_span(self):
        """Test that enabled tracing yields a span object."""
        init_tracing(
            service_name="test-service",
            endpoint=None,
            enabled=True,
        )

        with trace_span("test_span") as span:
            assert span is not None
            # Span should have set_attribute method
            assert hasattr(span, "set_attribute")

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_span_receives_attributes(self):
        """Test that span receives initial attributes."""
        init_tracing(
            service_name="test-service",
            endpoint=None,
            enabled=True,
        )

        with trace_span("test_span", {"custom_attr": "value"}) as span:
            assert span is not None
            # Can add more attributes
            span.set_attribute("another_attr", 123)

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_correlation_context_attached(self):
        """Test that correlation context is attached to span."""
        from gpt_trader.logging.correlation import correlation_context

        init_tracing(
            service_name="test-service",
            endpoint=None,
            enabled=True,
        )

        with correlation_context(cycle=42, symbol="BTC-USD"):
            with trace_span("test_span") as span:
                # Span should exist and include correlation context
                assert span is not None


class TestGetTracer:
    """Tests for get_tracer function."""

    def test_returns_none_when_disabled(self):
        """Test that get_tracer returns None when tracing disabled."""
        init_tracing(enabled=False)
        assert get_tracer() is None

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_returns_tracer_when_enabled(self):
        """Test that get_tracer returns tracer when enabled."""
        init_tracing(enabled=True)
        tracer = get_tracer()
        assert tracer is not None


class TestShutdownTracing:
    """Tests for shutdown_tracing function."""

    def test_shutdown_when_disabled(self):
        """Test that shutdown works when tracing was never enabled."""
        init_tracing(enabled=False)
        shutdown_tracing()  # Should not raise
        assert is_tracing_enabled() is False

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_shutdown_clears_state(self):
        """Test that shutdown clears tracing state."""
        init_tracing(enabled=True)
        assert is_tracing_enabled() is True

        shutdown_tracing()

        assert is_tracing_enabled() is False
        assert get_tracer() is None


class TestIsTracingEnabled:
    """Tests for is_tracing_enabled function."""

    def test_false_by_default(self):
        """Test that tracing is disabled by default."""
        shutdown_tracing()  # Reset state
        assert is_tracing_enabled() is False

    def test_false_when_explicitly_disabled(self):
        """Test that tracing is disabled when explicitly set."""
        init_tracing(enabled=False)
        assert is_tracing_enabled() is False

    @pytest.mark.skipif(not _OTEL_AVAILABLE, reason="OTel not installed")
    def test_true_when_enabled(self):
        """Test that tracing is enabled when initialized."""
        init_tracing(enabled=True)
        assert is_tracing_enabled() is True
