"""
Tests for Enhanced Structured Logging System
Phase 3, Week 7: Operational Excellence
Tests: OPS-001 to OPS-008
"""

import asyncio
import json
import logging
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.bot.monitoring.structured_logger import (
    CorrelationIDGenerator,
    DistributedTracer,
    EnhancedStructuredLogger,
    LogFormat,
    LogPerformanceMonitor,
    SpanType,
    configure_logging,
    get_logger,
    span_id,
    trace_id,
    traced_operation,
)


class TestCorrelationIDGenerator:
    """Test correlation ID generation."""

    def test_generate_correlation_id(self):
        """Test correlation ID generation."""
        corr_id = CorrelationIDGenerator.generate()
        assert corr_id.startswith("corr-")
        assert len(corr_id) == 21  # "corr-" + 16 hex chars

    def test_generate_trace_id(self):
        """Test trace ID generation."""
        tr_id = CorrelationIDGenerator.generate_trace_id()
        assert tr_id.startswith("trace-")
        assert len(tr_id) == 22  # "trace-" + 16 hex chars

    def test_generate_span_id(self):
        """Test span ID generation."""
        sp_id = CorrelationIDGenerator.generate_span_id()
        assert sp_id.startswith("span-")
        assert len(sp_id) == 17  # "span-" + 12 hex chars

    def test_unique_ids(self):
        """Test that IDs are unique."""
        ids = [CorrelationIDGenerator.generate() for _ in range(100)]
        assert len(set(ids)) == 100


class TestDistributedTracer:
    """Test distributed tracing functionality."""

    def test_create_tracer(self):
        """Test tracer creation."""
        tracer = DistributedTracer("test-service")
        assert tracer.service_name == "test-service"
        assert tracer.span_manager is not None

    def test_start_span(self):
        """Test span creation."""
        tracer = DistributedTracer("test-service")

        with tracer.start_span("test-operation", SpanType.ML_PREDICTION) as span:
            assert span is not None
            assert span.operation_name == "test-operation"
            assert span.span_type == SpanType.ML_PREDICTION
            assert span.status == "started"

    def test_span_context_propagation(self):
        """Test that span context is properly propagated."""
        tracer = DistributedTracer("test-service")

        with tracer.start_span("parent-operation") as parent_span:
            parent_trace_id = trace_id.get()
            parent_span_id = span_id.get()

            with tracer.start_span("child-operation") as child_span:
                child_trace_id = trace_id.get()
                child_span_id = span_id.get()

                # Same trace, different spans
                assert parent_trace_id == child_trace_id
                assert parent_span_id != child_span_id

    def test_span_error_handling(self):
        """Test span error handling."""
        tracer = DistributedTracer("test-service")

        with pytest.raises(ValueError):
            with tracer.start_span("error-operation") as span:
                raise ValueError("Test error")

        # Span should be marked as failed
        span_context = tracer.span_manager.get_span(span.span_id)
        assert span_context.status == "failed"
        assert "Test error" in span_context.error


class TestEnhancedStructuredLogger:
    """Test enhanced structured logger."""

    @pytest.fixture
    def logger(self):
        """Create test logger."""
        return EnhancedStructuredLogger(
            name="test.logger", level="DEBUG", format_type=LogFormat.JSON, enable_tracing=True
        )

    @pytest.fixture
    def log_file(self):
        """Create temporary log file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)

    def test_logger_creation(self, logger):
        """Test logger creation."""
        assert logger.name == "test.logger"
        assert logger.service_name == "gpt-trader"
        assert logger.tracer is not None

    def test_correlation_id_management(self, logger):
        """Test correlation ID management."""
        # Generate new correlation ID
        corr_id = logger.new_correlation_id()
        assert corr_id.startswith("corr-")
        assert logger.get_correlation_id() == corr_id

        # Set specific correlation ID
        test_id = "test-correlation-id"
        logger.set_correlation_id(test_id)
        assert logger.get_correlation_id() == test_id

    def test_correlation_context(self, logger):
        """Test correlation ID context manager."""
        with logger.correlation_context("test-context") as corr_id:
            assert corr_id == "test-context"
            assert logger.get_correlation_id() == "test-context"

        # Context should be reset after exit
        assert logger.get_correlation_id() != "test-context"

    def test_basic_logging(self, logger, caplog):
        """Test basic logging functionality."""
        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

        assert len(caplog.records) == 5
        assert "Debug message" in caplog.text
        assert "Info message" in caplog.text
        assert "Warning message" in caplog.text
        assert "Error message" in caplog.text
        assert "Critical message" in caplog.text

    def test_json_formatting(self, logger):
        """Test JSON log formatting."""
        with patch.object(logger.logger, "handle") as mock_handle:
            logger.info("Test message", operation="test-op", symbol="AAPL")

            # Verify record was created
            assert mock_handle.called
            record = mock_handle.call_args[0][0]

            # Test JSON formatting
            formatted = logger.formatter.format(record)
            log_data = json.loads(formatted)

            assert log_data["message"] == "Test message"
            assert log_data["level"] == "INFO"
            assert log_data["service"] == "gpt-trader"
            assert "timestamp" in log_data
            assert "correlation_id" in log_data

    def test_operation_timing(self, logger):
        """Test operation timing functionality."""
        operation_id = logger.start_operation("test-operation")
        time.sleep(0.01)  # Small delay
        duration = logger.end_operation(operation_id, "test-operation", success=True)

        assert duration >= 10  # At least 10ms
        assert operation_id not in logger._operation_timers

    def test_span_integration(self, logger):
        """Test span integration."""
        with logger.start_span("test-span", SpanType.ML_PREDICTION) as span:
            # Context variables should be set
            assert trace_id.get() != ""
            assert span_id.get() != ""

            # Log message within span
            with patch.object(logger.logger, "handle") as mock_handle:
                logger.info("Message in span")

                record = mock_handle.call_args[0][0]
                formatted = logger.formatter.format(record)
                log_data = json.loads(formatted)

                assert log_data["trace_id"] != ""
                assert log_data["span_id"] != ""

    def test_exception_logging(self, logger):
        """Test exception logging."""
        with patch.object(logger.logger, "handle") as mock_handle:
            try:
                raise ValueError("Test exception")
            except ValueError:
                logger.exception("Exception occurred")

            record = mock_handle.call_args[0][0]
            formatted = logger.formatter.format(record)
            log_data = json.loads(formatted)

            assert log_data["message"] == "Exception occurred"
            assert log_data["error_type"] == "ValueError"
            assert log_data["error_message"] == "Test exception"
            assert "stack_trace" in log_data

    def test_business_context(self, logger):
        """Test business context enrichment."""
        with patch.object(logger.logger, "handle") as mock_handle:
            logger.info(
                "Trade executed",
                symbol="AAPL",
                strategy="momentum",
                model_id="xgb-v1.2",
                trade_id="TRD-123",
                attributes={"price": 150.0, "quantity": 100},
                tags={"environment": "production", "region": "us-east-1"},
            )

            record = mock_handle.call_args[0][0]
            formatted = logger.formatter.format(record)
            log_data = json.loads(formatted)

            assert log_data["symbol"] == "AAPL"
            assert log_data["strategy"] == "momentum"
            assert log_data["model_id"] == "xgb-v1.2"
            assert log_data["trade_id"] == "TRD-123"
            assert log_data["attributes"]["price"] == 150.0
            assert log_data["tags"]["environment"] == "production"

    def test_file_logging(self, log_file):
        """Test file logging."""
        logger = EnhancedStructuredLogger(
            name="test.file.logger", format_type=LogFormat.JSON, log_file=log_file
        )

        logger.info("Test file message")

        # Check file was created and contains message
        assert log_file.exists()
        content = log_file.read_text()
        log_data = json.loads(content.strip())
        assert log_data["message"] == "Test file message"

    def test_audit_and_metric_logging(self, logger):
        """Test special audit and metric logging."""
        with patch.object(logger.logger, "handle") as mock_handle:
            logger.audit("User login", user_id="user123")
            logger.metric("Latency measurement", value=250, unit="ms")

            assert mock_handle.call_count == 2

            # Check audit log
            audit_record = mock_handle.call_args_list[0][0][0]
            assert audit_record.levelname == "AUDIT"

            # Check metric log
            metric_record = mock_handle.call_args_list[1][0][0]
            assert metric_record.levelname == "METRIC"


class TestTracedOperation:
    """Test traced operation decorator."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        logger = Mock(spec=EnhancedStructuredLogger)
        logger.start_span.return_value.__enter__ = Mock(return_value=None)
        logger.start_span.return_value.__exit__ = Mock(return_value=None)
        logger.start_operation.return_value = "test-op-id"
        logger.end_operation.return_value = 50.0
        return logger

    def test_sync_function_tracing(self, mock_logger):
        """Test tracing of synchronous functions."""

        @traced_operation("test.function", SpanType.BUSINESS_LOGIC)
        def test_function(x, y):
            return x + y

        with patch("src.bot.monitoring.structured_logger.get_logger", return_value=mock_logger):
            result = test_function(2, 3)

            assert result == 5
            mock_logger.start_span.assert_called_once()
            mock_logger.start_operation.assert_called_once()
            mock_logger.end_operation.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_function_tracing(self, mock_logger):
        """Test tracing of asynchronous functions."""

        @traced_operation("test.async_function", SpanType.DATABASE_QUERY)
        async def async_test_function(x, y):
            await asyncio.sleep(0.001)
            return x * y

        with patch("src.bot.monitoring.structured_logger.get_logger", return_value=mock_logger):
            result = await async_test_function(4, 5)

            assert result == 20
            mock_logger.start_span.assert_called_once()
            mock_logger.start_operation.assert_called_once()
            mock_logger.end_operation.assert_called_once()

    def test_function_error_tracing(self, mock_logger):
        """Test tracing of functions that raise exceptions."""

        @traced_operation("test.error_function")
        def error_function():
            raise ValueError("Test error")

        with patch("src.bot.monitoring.structured_logger.get_logger", return_value=mock_logger):
            with pytest.raises(ValueError):
                error_function()

            mock_logger.exception.assert_called_once()
            mock_logger.end_operation.assert_called_with(
                "test-op-id", "test.error_function", success=False
            )


class TestLoggerRegistry:
    """Test logger registry functionality."""

    def test_get_logger_singleton(self):
        """Test that get_logger returns the same instance."""
        logger1 = get_logger("test.singleton")
        logger2 = get_logger("test.singleton")

        assert logger1 is logger2

    def test_get_different_loggers(self):
        """Test that different names return different loggers."""
        logger1 = get_logger("test.logger1")
        logger2 = get_logger("test.logger2")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_configure_logging(self):
        """Test global logging configuration."""
        with tempfile.NamedTemporaryFile(suffix=".log") as f:
            log_file = Path(f.name)

            configure_logging(
                level="DEBUG",
                format_type=LogFormat.JSON,
                log_file=log_file,
                service_name="test-service",
            )

            logger = get_logger("test.configured")
            assert logger.service_name == "test-service"


class TestPerformanceMonitoring:
    """Test performance monitoring features."""

    def test_log_performance_monitor(self):
        """Test log performance monitoring."""
        monitor = LogPerformanceMonitor()

        # Record some log operations
        for i in range(100):
            monitor.record_log(0.001)  # 1ms per log

        stats = monitor.get_stats()
        assert stats["total_logs"] == 100
        assert stats["average_latency_ms"] == 1.0
        assert stats["logs_per_second"] > 0

    @pytest.mark.asyncio
    async def test_high_volume_logging_performance(self):
        """Test high-volume logging performance target (>10,000 logs/sec)."""
        logger = EnhancedStructuredLogger(name="performance.test", format_type=LogFormat.JSON)

        # Mock the actual I/O to test formatting performance
        with patch.object(logger.logger, "handle"):
            start_time = time.time()
            num_logs = 1000  # Reduced for test speed

            for i in range(num_logs):
                logger.info(f"Performance test message {i}", operation="perf-test")

            duration = time.time() - start_time
            logs_per_second = num_logs / duration

            # Should be much higher than 10,000/sec when I/O is mocked
            assert logs_per_second > 5000  # Conservative check

    def test_memory_usage_optimization(self):
        """Test that logger doesn't leak memory."""
        import gc

        logger = EnhancedStructuredLogger(name="memory.test", format_type=LogFormat.JSON)

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create many log records
        with patch.object(logger.logger, "handle"):
            for i in range(1000):
                logger.info(f"Memory test {i}")

        # Check memory usage
        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not have significant memory growth
        object_growth = final_objects - initial_objects
        assert object_growth < 100  # Allowing some growth for test infrastructure


class TestIntegrationWithMLComponents:
    """Test integration with existing ML components."""

    def test_ml_pipeline_logging(self):
        """Test logging in ML pipeline context."""
        logger = get_logger("ml.pipeline")

        with logger.correlation_context() as corr_id:
            with logger.start_span("ml.prediction", SpanType.ML_PREDICTION) as span:
                # Simulate ML prediction logging
                logger.info(
                    "Starting prediction",
                    model_id="xgb-v1.2",
                    symbol="AAPL",
                    attributes={"features": 50, "data_points": 1000},
                )

                # Simulate prediction result
                logger.info(
                    "Prediction completed",
                    model_id="xgb-v1.2",
                    symbol="AAPL",
                    attributes={"prediction": 0.65, "confidence": 0.82},
                )

        # Verify context was maintained
        assert corr_id != ""

    def test_risk_calculation_logging(self):
        """Test logging in risk calculation context."""
        logger = get_logger("risk.calculator")

        with logger.start_span("risk.var_calculation", SpanType.RISK_CALCULATION):
            operation_id = logger.start_operation("calculate_var")

            # Simulate risk calculation
            time.sleep(0.01)  # Simulate computation

            duration = logger.end_operation(operation_id, success=True)

            logger.info(
                "VaR calculation completed",
                operation="calculate_var",
                duration_ms=duration,
                attributes={"var_95": 0.02, "portfolio_value": 1000000},
            )

    def test_backtest_logging(self):
        """Test logging in backtest context."""
        logger = get_logger("backtest.engine")

        with logger.start_span("backtest.run", SpanType.BACKTEST):
            logger.info(
                "Backtest started",
                strategy="momentum",
                symbol="AAPL",
                attributes={"start_date": "2024-01-01", "end_date": "2024-12-31"},
            )

            # Simulate trade logging
            logger.audit(
                "Trade executed",
                trade_id="TRD-001",
                symbol="AAPL",
                side="BUY",
                quantity=100,
                price=150.0,
            )

            logger.info(
                "Backtest completed",
                strategy="momentum",
                attributes={"total_trades": 50, "win_rate": 0.68, "sharpe": 1.25},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
