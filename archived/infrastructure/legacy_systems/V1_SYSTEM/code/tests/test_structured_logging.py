"""Unit tests for structured logging system."""

import json
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from bot.logging import (
    LogAggregator,
    LogEntry,
    LogFormat,
    PerformanceLogger,
    StructuredFormatter,
    StructuredLogger,
    TradeLogger,
)


class TestStructuredFormatter:
    """Test StructuredFormatter class."""

    def test_json_format(self):
        """Test JSON formatting."""
        formatter = StructuredFormatter(format_type=LogFormat.JSON)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["level"] == "INFO"
        assert data["logger"] == "test.logger"
        assert data["message"] == "Test message"
        assert data["line"] == 10
        assert "timestamp" in data

    def test_json_format_with_exception(self):
        """Test JSON formatting with exception."""
        formatter = StructuredFormatter(format_type=LogFormat.JSON)

        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error"
        assert data["exception"]["traceback"] is not None

    def test_text_format(self):
        """Test text formatting."""
        formatter = StructuredFormatter(format_type=LogFormat.TEXT)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)

        assert "test.logger" in formatted
        assert "INFO" in formatted
        assert "Test message" in formatted

    def test_context_fields(self):
        """Test context fields in formatting."""
        context_fields = {"request_id": "123", "user": "test_user"}
        formatter = StructuredFormatter(format_type=LogFormat.JSON, context_fields=context_fields)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["request_id"] == "123"
        assert data["user"] == "test_user"


class TestTradeLogger:
    """Test TradeLogger class."""

    def test_log_order_placed(self):
        """Test logging order placement."""
        mock_logger = Mock()
        trade_logger = TradeLogger(mock_logger)

        trade_logger.log_order_placed(
            symbol="AAPL", side="buy", quantity=100, price=150.0, order_type="limit"
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args

        assert "Order placed" in call_args[0][0]
        assert "AAPL" in call_args[0][0]

        extra_fields = call_args[1]["extra"]["extra_fields"]
        assert extra_fields["event_type"] == "order_placed"
        assert extra_fields["symbol"] == "AAPL"
        assert extra_fields["side"] == "buy"
        assert extra_fields["quantity"] == 100
        assert extra_fields["price"] == 150.0

    def test_log_position_closed(self):
        """Test logging position closing."""
        mock_logger = Mock()
        trade_logger = TradeLogger(mock_logger)

        trade_logger.log_position_closed(
            symbol="GOOGL", quantity=50, entry_price=2000.0, exit_price=2100.0, pnl=5000.0
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args

        assert "Position closed" in call_args[0][0]
        assert "GOOGL" in call_args[0][0]
        assert "5000.00" in call_args[0][0]

        extra_fields = call_args[1]["extra"]["extra_fields"]
        assert extra_fields["event_type"] == "position_closed"
        assert extra_fields["pnl"] == 5000.0
        assert extra_fields["return_pct"] == 5.0  # (2100-2000)/2000 * 100


class TestPerformanceLogger:
    """Test PerformanceLogger class."""

    def test_log_metric(self):
        """Test logging metrics."""
        mock_logger = Mock()
        perf_logger = PerformanceLogger(mock_logger)

        perf_logger.log_metric(
            metric_name="sharpe_ratio", value=1.5, unit="", tags={"strategy": "trend_following"}
        )

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args

        assert "sharpe_ratio=1.5" in call_args[0][0]

        extra_fields = call_args[1]["extra"]["extra_fields"]
        assert extra_fields["event_type"] == "metric"
        assert extra_fields["metric_name"] == "sharpe_ratio"
        assert extra_fields["value"] == 1.5
        assert extra_fields["tags"]["strategy"] == "trend_following"

    def test_log_latency(self):
        """Test logging latency."""
        mock_logger = Mock()
        perf_logger = PerformanceLogger(mock_logger)

        # Normal latency
        perf_logger.log_latency(operation="data_fetch", latency_ms=50.0, threshold_ms=100.0)

        mock_logger.log.assert_called_with(
            logging.INFO,
            "Latency: data_fetch took 50.00ms",
            extra={
                "extra_fields": {
                    "event_type": "latency",
                    "operation": "data_fetch",
                    "latency_ms": 50.0,
                    "threshold_ms": 100.0,
                    "slow": False,
                }
            },
        )

        # Slow latency
        perf_logger.log_latency(operation="backtest", latency_ms=200.0, threshold_ms=100.0)

        # Second call should be WARNING level
        assert mock_logger.log.call_args[0][0] == logging.WARNING


class TestStructuredLogger:
    """Test StructuredLogger class."""

    def test_initialization(self):
        """Test logger initialization."""
        with patch("bot.logging.structured_logger.get_config") as mock_config:
            mock_config.return_value.logging.level = "INFO"
            mock_config.return_value.logging.structured_logging = True
            mock_config.return_value.logging.file_path = None

            logger = StructuredLogger("test_logger")

            assert logger.logger.name == "test_logger"
            assert logger.logger.level == logging.INFO
            assert isinstance(logger.trade, TradeLogger)
            assert isinstance(logger.performance, PerformanceLogger)

    def test_add_context(self):
        """Test adding context fields."""
        with patch("bot.logging.structured_logger.get_config") as mock_config:
            mock_config.return_value.logging.level = "INFO"
            mock_config.return_value.logging.structured_logging = True
            mock_config.return_value.logging.file_path = None

            logger = StructuredLogger("test_logger")

            logger.add_context(request_id="123", user="test_user")

            assert logger.formatter.context_fields["request_id"] == "123"
            assert logger.formatter.context_fields["user"] == "test_user"

    def test_with_context(self):
        """Test context manager for temporary context."""
        with patch("bot.logging.structured_logger.get_config") as mock_config:
            mock_config.return_value.logging.level = "INFO"
            mock_config.return_value.logging.structured_logging = True
            mock_config.return_value.logging.file_path = None

            logger = StructuredLogger("test_logger")

            # Add permanent context
            logger.add_context(app="trading")

            # Use temporary context
            with logger.with_context(request_id="456"):
                assert logger.formatter.context_fields["request_id"] == "456"
                assert logger.formatter.context_fields["app"] == "trading"

            # Temporary context should be removed
            assert "request_id" not in logger.formatter.context_fields
            assert logger.formatter.context_fields["app"] == "trading"

    def test_file_logging(self):
        """Test logging to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            with patch("bot.logging.structured_logger.get_config") as mock_config:
                mock_config.return_value.logging.level = "INFO"
                mock_config.return_value.logging.structured_logging = False
                mock_config.return_value.logging.file_path = log_file

                logger = StructuredLogger("test_logger", log_file=log_file)
                logger.info("Test message")

                # Check file was created and contains log
                assert log_file.exists()
                content = log_file.read_text()
                assert "Test message" in content


class TestLogEntry:
    """Test LogEntry class."""

    def test_from_json(self):
        """Test creating LogEntry from JSON."""
        json_str = json.dumps(
            {
                "timestamp": "2024-01-01T12:00:00",
                "level": "INFO",
                "logger": "test.logger",
                "message": "Test message",
                "module": "test_module",
                "function": "test_func",
                "line": 42,
                "custom_field": "custom_value",
            }
        )

        entry = LogEntry.from_json(json_str)

        assert entry.level == "INFO"
        assert entry.logger == "test.logger"
        assert entry.message == "Test message"
        assert entry.module == "test_module"
        assert entry.function == "test_func"
        assert entry.line == 42
        assert entry.extra_fields["custom_field"] == "custom_value"

    def test_from_text(self):
        """Test creating LogEntry from text."""
        text_str = "2024-01-01 12:00:00,123 | test.logger | INFO | Test message"

        entry = LogEntry.from_text(text_str)

        assert entry.level == "INFO"
        assert entry.logger == "test.logger"
        assert entry.message == "Test message"


class TestLogAggregator:
    """Test LogAggregator class."""

    def test_add_entry(self):
        """Test adding entries to aggregator."""
        aggregator = LogAggregator()

        entry = LogEntry(
            timestamp=datetime.now(), level="ERROR", logger="test", message="Error message"
        )

        aggregator.add_entry(entry)

        assert aggregator.stats["total_entries"] == 1
        assert aggregator.stats["errors"] == 1
        assert len(aggregator.errors) == 1

    def test_trade_event_categorization(self):
        """Test categorizing trade events."""
        aggregator = LogAggregator()

        entry = LogEntry(
            timestamp=datetime.now(),
            level="INFO",
            logger="trade",
            message="Order placed",
            extra_fields={"event_type": "order_placed", "symbol": "AAPL"},
        )

        aggregator.add_entry(entry)

        assert aggregator.stats["trades"] == 1
        assert len(aggregator.trades) == 1

    def test_get_error_summary(self):
        """Test error summary generation."""
        aggregator = LogAggregator()

        # Add some errors
        for i in range(5):
            entry = LogEntry(
                timestamp=datetime.now(),
                level="ERROR",
                logger="test",
                message=f"Error {i}",
                module=f"module_{i % 2}",
                exception={"type": "ValueError" if i % 2 == 0 else "TypeError"},
            )
            aggregator.add_entry(entry)

        summary = aggregator.get_error_summary()

        assert summary["total_errors"] == 5
        assert summary["error_types"]["ValueError"] == 3
        assert summary["error_types"]["TypeError"] == 2
        assert len(summary["recent_errors"]) <= 10

    def test_get_trade_summary(self):
        """Test trade summary generation."""
        aggregator = LogAggregator()

        # Add trade events
        events = [
            ("order_placed", {"symbol": "AAPL"}),
            ("order_filled", {"symbol": "AAPL"}),
            ("position_opened", {"symbol": "GOOGL"}),
            ("position_closed", {"symbol": "GOOGL", "pnl": 500.0}),
        ]

        for event_type, extra in events:
            entry = LogEntry(
                timestamp=datetime.now(),
                level="INFO",
                logger="trade",
                message=f"{event_type}",
                extra_fields={"event_type": event_type, **extra},
            )
            aggregator.add_entry(entry)

        summary = aggregator.get_trade_summary()

        assert summary["total_trades"] == 4
        assert summary["orders_placed"] == 1
        assert summary["orders_filled"] == 1
        assert summary["positions_closed"] == 1
        assert summary["total_pnl"] == 500.0
        assert set(summary["symbols_traded"]) == {"AAPL", "GOOGL"}

    def test_detect_patterns(self):
        """Test pattern detection."""
        aggregator = LogAggregator()

        # Add error burst
        base_time = datetime.now()
        for i in range(6):
            entry = LogEntry(
                timestamp=base_time + timedelta(seconds=i),
                level="ERROR",
                logger="test",
                message="Repeated error message",
            )
            aggregator.add_entry(entry)

        patterns = aggregator.detect_patterns()

        # Should detect error burst
        assert len(patterns["error_bursts"]) > 0
        assert patterns["error_bursts"][0]["count"] >= 5

        # Should detect repeated errors
        assert len(patterns["repeated_errors"]) > 0

    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        aggregator = LogAggregator()

        # Add various entries
        for i in range(5):
            entry = LogEntry(
                timestamp=datetime.now(),
                level="INFO",
                logger=f"logger_{i}",
                message=f"Message {i}",
                extra_fields={"index": i},
            )
            aggregator.add_entry(entry)

        df = aggregator.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "timestamp" in df.columns
        assert "level" in df.columns
        assert "extra_index" in df.columns

    def test_generate_report(self):
        """Test report generation."""
        aggregator = LogAggregator()

        # Add various log entries
        for i in range(10):
            level = "ERROR" if i < 3 else "INFO"
            entry = LogEntry(
                timestamp=datetime.now(), level=level, logger="test", message=f"{level} message {i}"
            )
            aggregator.add_entry(entry)

        report = aggregator.generate_report()

        assert "LOG ANALYSIS REPORT" in report
        assert "Total Entries: 10" in report
        assert "Errors: 3" in report
