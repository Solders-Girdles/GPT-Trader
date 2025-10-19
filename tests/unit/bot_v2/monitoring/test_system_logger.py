"""Tests for system logger serializer and level handling."""

from __future__ import annotations

import logging
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from bot_v2.monitoring.system.logger import (
    LogLevel,
    ProductionLogger,
    get_correlation_id,
    get_logger,
    log_error,
    log_event,
    log_ml_prediction,
    log_performance,
    log_trade,
    set_correlation_id,
)


class TestProductionLoggerSerializer:
    """Test JSON serialization and level handling."""

    @pytest.fixture
    def logger(self):
        """Create a test logger with console disabled."""
        return ProductionLogger("test_service", enable_console=False)

    def test_log_entry_json_serialization(self, logger):
        """Test that log entries are properly serialized to JSON."""
        with patch.object(logger, "_emit_log") as mock_emit:
            logger.log_event(
                LogLevel.INFO,
                "test_event",
                "test message",
                component="test_component",
                extra_field="extra_value",
                numeric_field=42,
                decimal_field=Decimal("1.23"),
            )

            # Verify _emit_log was called with properly serialized entry
            mock_emit.assert_called_once()
            entry = mock_emit.call_args[0][0]

            # Check required fields
            assert entry["level"] == "info"
            assert entry["service"] == "test_service"
            assert entry["event_type"] == "test_event"
            assert entry["message"] == "test message"
            assert entry["component"] == "test_component"
            assert entry["correlation_id"] is not None

            # Check extra fields are preserved
            assert entry["extra_field"] == "extra_value"
            assert entry["numeric_field"] == 42
            assert entry["decimal_field"] == "1.23"  # Decimal converted to string

            # Check timestamp format
            assert "timestamp" in entry
            assert entry["timestamp"].endswith("Z")  # ISO format with Z

    def test_log_level_filtering(self, logger):
        """Test that log levels are properly filtered."""
        logger._min_level = "warning"

        with patch.object(logger, "_emit_log") as mock_emit:
            # Should be filtered out
            logger.log_event(LogLevel.INFO, "test", "info message")
            logger.log_event(LogLevel.DEBUG, "test", "debug message")

            # Should be emitted
            logger.log_event(LogLevel.WARNING, "test", "warning message")
            logger.log_event(LogLevel.ERROR, "test", "error message")

            # Verify only warning+ levels were emitted
            assert mock_emit.call_count == 2
            levels = [call[0][0]["level"] for call in mock_emit.call_args_list]
            assert "warning" in levels
            assert "error" in levels
            assert "info" not in levels
            assert "debug" not in levels

    def test_log_level_debug_override(self):
        """Test PERPS_DEBUG environment variable overrides min level."""
        with patch.dict("os.environ", {"PERPS_DEBUG": "1"}):
            logger = ProductionLogger("test", enable_console=False)
            assert logger._min_level == "debug"

    def test_log_level_env_override(self):
        """Test PERPS_MIN_LOG_LEVEL environment variable."""
        with patch.dict("os.environ", {"PERPS_MIN_LOG_LEVEL": "error"}):
            logger = ProductionLogger("test", enable_console=False)
            assert logger._min_level == "error"

    def test_console_output_control(self):
        """Test console output can be disabled via environment."""
        with patch.dict("os.environ", {"PERPS_JSON_CONSOLE": "0"}):
            logger = ProductionLogger("test", enable_console=True)
            assert logger.enable_console is False

        with patch.dict("os.environ", {"PERPS_JSON_CONSOLE": "true"}):
            logger = ProductionLogger("test", enable_console=False)
            assert logger.enable_console is True

    def test_correlation_id_thread_isolation(self):
        """Test correlation IDs are isolated per thread."""
        logger1 = ProductionLogger("test1", enable_console=False)
        logger2 = ProductionLogger("test2", enable_console=False)

        # Set different correlation IDs
        logger1.set_correlation_id("thread1")
        logger2.set_correlation_id("thread2")

        assert logger1.get_correlation_id() == "thread1"
        assert logger2.get_correlation_id() == "thread2"

    def test_correlation_id_auto_generation(self, logger):
        """Test correlation ID auto-generation."""
        # Clear any existing ID
        if hasattr(logger.correlation_ids, "value"):
            delattr(logger.correlation_ids, "value")

        correlation_id = logger.get_correlation_id()

        # Should be 8 characters (UUID prefix)
        assert len(correlation_id) == 8
        assert correlation_id.isalnum()

    def test_recent_logs_buffer(self, logger):
        """Test recent logs buffer management."""
        # Fill buffer beyond limit
        logger._max_recent_logs = 3

        for i in range(5):
            logger._emit_log({"test": f"log_{i}"})

        recent = logger.get_recent_logs()
        assert len(recent) == 3
        # Should contain the last 3 logs
        assert recent[0]["test"] == "log_2"
        assert recent[1]["test"] == "log_3"
        assert recent[2]["test"] == "log_4"

    def test_performance_stats_tracking(self, logger):
        """Test performance statistics tracking."""
        # Simulate some log operations
        for _ in range(3):
            logger._create_log_entry(LogLevel.INFO, "test", "message")

        stats = logger.get_performance_stats()
        assert stats["total_logs"] == 3
        assert "avg_log_time_ms" in stats
        assert "total_log_time_ms" in stats
        assert stats["avg_log_time_ms"] > 0

    def test_empty_performance_stats(self, logger):
        """Test performance stats with no logs."""
        stats = logger.get_performance_stats()
        assert stats["total_logs"] == 0
        assert stats["avg_log_time_ms"] == 0.0


class TestSpecializedLogMethods:
    """Test specialized logging methods with proper serialization."""

    @pytest.fixture
    def logger(self):
        return ProductionLogger("test", enable_console=False)

    def test_log_trade_success_serialization(self, logger):
        """Test trade logging with success case."""
        with patch.object(logger, "_emit_log") as mock_emit:
            logger.log_trade(
                action="buy",
                symbol="BTC-USD",
                quantity=1.5,
                price=50000.0,
                strategy="test_strategy",
                success=True,
                execution_time_ms=150.5,
                client_order_id="test_123",
            )

            entry = mock_emit.call_args[0][0]
            assert entry["level"] == "info"
            assert entry["event_type"] == "trade_execution"
            assert entry["message"] == "BUY 1.5 BTC-USD @ 50000.0"
            assert entry["trade_action"] == "buy"
            assert entry["symbol"] == "BTC-USD"
            assert entry["quantity"] == 1.5
            assert entry["price"] == 50000.0
            assert entry["strategy"] == "test_strategy"
            assert entry["success"] is True
            assert entry["execution_time_ms"] == 150.5
            assert entry["client_order_id"] == "test_123"

    def test_log_trade_failure_serialization(self, logger):
        """Test trade logging with failure case."""
        with patch.object(logger, "_emit_log") as mock_emit:
            logger.log_trade(
                action="sell",
                symbol="ETH-USD",
                quantity=10.0,
                price=3000.0,
                strategy="test_strategy",
                success=False,
                error_code="INSUFFICIENT_FUNDS",
            )

            entry = mock_emit.call_args[0][0]
            assert entry["level"] == "error"
            assert entry["event_type"] == "trade_execution"
            assert entry["success"] is False
            assert entry["error_code"] == "INSUFFICIENT_FUNDS"

    def test_log_ml_prediction_serialization(self, logger):
        """Test ML prediction logging with feature summarization."""
        with patch.object(logger, "_emit_log") as mock_emit:
            features = {f"feature_{i}": i for i in range(10)}  # 10 features

            logger.log_ml_prediction(
                model_name="test_model",
                prediction="BUY",
                confidence=0.85,
                input_features=features,
                inference_time_ms=25.3,
                model_version="v1.2.3",
            )

            entry = mock_emit.call_args[0][0]
            assert entry["event_type"] == "ml_prediction"
            assert entry["model_name"] == "test_model"
            assert entry["prediction"] == "BUY"
            assert entry["confidence"] == 0.85
            assert entry["inference_time_ms"] == 25.3
            assert entry["model_version"] == "v1.2.3"

            # Check feature summarization
            assert entry["feature_count"] == 10
            assert "sample_features" in entry
            assert len(entry["sample_features"]) == 5  # Limited to first 5

    def test_log_performance_serialization(self, logger):
        """Test performance logging."""
        with patch.object(logger, "_emit_log") as mock_emit:
            logger.log_performance(
                operation="database_query",
                duration_ms=45.2,
                success=True,
                query_type="SELECT",
                row_count=100,
            )

            entry = mock_emit.call_args[0][0]
            assert entry["level"] == "info"
            assert entry["event_type"] == "performance_metric"
            assert entry["operation"] == "database_query"
            assert entry["duration_ms"] == 45.2
            assert entry["success"] is True
            assert entry["query_type"] == "SELECT"
            assert entry["row_count"] == 100

    def test_log_performance_failure_serialization(self, logger):
        """Test performance logging with failure."""
        with patch.object(logger, "_emit_log") as mock_emit:
            logger.log_performance(
                operation="api_call",
                duration_ms=5000.0,
                success=False,
                error_type="TimeoutError",
            )

            entry = mock_emit.call_args[0][0]
            assert entry["level"] == "warning"
            assert entry["success"] is False

    def test_log_error_serialization(self, logger):
        """Test error logging with exception details."""
        with patch.object(logger, "_emit_log") as mock_emit:
            try:
                raise ValueError("Test error message")
            except ValueError as e:
                logger.log_error(e, context="test_operation", user_id="user123")

            entry = mock_emit.call_args[0][0]
            assert entry["level"] == "error"
            assert entry["event_type"] == "error"
            assert entry["message"] == "Test error message"
            assert entry["error_type"] == "ValueError"
            assert entry["error_context"] == "test_operation"
            assert entry["user_id"] == "user123"

    def test_domain_specific_log_serialization(self, logger):
        """Test domain-specific logging methods."""
        with patch.object(logger, "_emit_log") as mock_emit:
            # Test PnL logging
            logger.log_pnl(
                symbol="BTC-USD",
                realized_pnl=100.5,
                unrealized_pnl=-25.3,
                fees=5.0,
                position_size=1.5,
            )

            pnl_entry = mock_emit.call_args[0][0]
            assert pnl_entry["event_type"] == "pnl_update"
            assert pnl_entry["symbol"] == "BTC-USD"
            assert pnl_entry["realized_pnl"] == 100.5
            assert pnl_entry["unrealized_pnl"] == -25.3

            # Test order submission
            logger.log_order_submission(
                client_order_id="client_123",
                symbol="ETH-USD",
                side="buy",
                order_type="limit",
                quantity=5.0,
                price=3000.0,
            )

            order_entry = mock_emit.call_args[0][0]
            assert order_entry["event_type"] == "order_submission"
            assert order_entry["client_order_id"] == "client_123"
            assert order_entry["symbol"] == "ETH-USD"
            assert order_entry["side"] == "buy"
            assert order_entry["quantity"] == 5.0
            assert order_entry["price"] == 3000.0

    def test_log_level_enum_mapping(self, logger):
        """Test LogLevel enum maps correctly to Python logging levels."""
        from bot_v2.monitoring.system.logger import _LEVEL_MAP

        assert _LEVEL_MAP["debug"] == logging.DEBUG
        assert _LEVEL_MAP["info"] == logging.INFO
        assert _LEVEL_MAP["warning"] == logging.WARNING
        assert _LEVEL_MAP["error"] == logging.ERROR
        assert _LEVEL_MAP["critical"] == logging.CRITICAL

    def test_python_logger_fallback(self, logger):
        """Test fallback to global logger when sub-logger has no handlers."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.handlers = []  # No handlers
            mock_get_logger.return_value = mock_logger

            # Create logger which should trigger fallback
            test_logger = ProductionLogger("test", enable_console=False)

            # Mock the fallback logger
            fallback_logger = Mock()
            mock_get_logger.side_effect = lambda name: (
                fallback_logger if "bot_v2.json" in str(name) else mock_logger
            )

            with patch.object(test_logger, "_emit_log"):
                test_logger.log_event(LogLevel.INFO, "test", "message")

                # Verify it uses the fallback logger
                fallback_logger.log.assert_called()


class TestGlobalLoggerFunctions:
    """Test global logger convenience functions."""

    def test_global_logger_singleton(self):
        """Test global logger is a singleton."""
        logger1 = get_logger("test1")
        logger2 = get_logger("test2")

        # Should return the same instance
        assert logger1 is logger2
        assert logger1.service_name == "test1"  # First call sets the name

    def test_convenience_functions(self):
        """Test convenience logging functions."""
        with patch("bot_v2.monitoring.system.logger.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # Test each convenience function
            log_event("test_event", "test message", level=LogLevel.WARNING, extra="value")
            mock_logger.log_event.assert_called_with(
                LogLevel.WARNING, "test_event", "test message", extra="value"
            )

            log_trade("buy", "BTC-USD", 1.0, 50000.0, "strategy")
            mock_logger.log_trade.assert_called_with("buy", "BTC-USD", 1.0, 50000.0, "strategy")

            log_ml_prediction("model", "prediction")
            mock_logger.log_ml_prediction.assert_called_with("model", "prediction")

            log_performance("operation", 100.0)
            mock_logger.log_performance.assert_called_with("operation", 100.0)

            log_error(ValueError("test"))
            mock_logger.log_error.assert_called_with(ValueError("test"), None)

            set_correlation_id("test_id")
            mock_logger.set_correlation_id.assert_called_with("test_id")

            mock_logger.get_correlation_id.return_value = "test_id"
            assert get_correlation_id() == "test_id"


class TestFakeAlertHandlers:
    """Test fake alert handlers for testing purposes."""

    def test_fake_log_handler(self):
        """Test fake log handler captures alerts properly."""
        from bot_v2.monitoring.guards.manager import log_alert_handler

        alert = Mock()
        alert.guard_name = "test_guard"
        alert.severity.value = "ERROR"
        alert.to_dict.return_value = {"test": "data"}

        with patch("bot_v2.monitoring.guards.manager.logger") as mock_logger:
            log_alert_handler(alert)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert "test_guard" in call_args[0][0]
            assert "ERROR" in call_args[0][0]
            # Check JSON payload is included
            assert '"test": "data"' in call_args[0][0]

    def test_fake_slack_handler_success(self):
        """Test fake Slack handler with successful request."""
        from bot_v2.monitoring.guards.manager import slack_alert_handler

        alert = Mock()
        alert.guard_name = "test_guard"
        alert.severity.value = "CRITICAL"

        with patch("bot_v2.monitoring.guards.manager.requests.post") as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            # Should not raise exception
            slack_alert_handler(alert, "https://hooks.slack.com/test")

            mock_post.assert_called_once()

    def test_fake_email_handler_filtering(self):
        """Test fake email handler filters non-critical alerts."""
        from bot_v2.monitoring.guards.manager import email_alert_handler

        # Non-critical alert should be filtered
        alert = Mock()
        alert.severity.value = "WARNING"

        with patch("bot_v2.monitoring.guards.manager.smtplib.SMTP") as mock_smtp:
            email_alert_handler(alert, {"host": "test", "port": 587, "from": "a", "to": "b"})

            # Should not create SMTP connection
            mock_smtp.assert_not_called()
