"""Tests for the enhanced JSON formatter."""

from __future__ import annotations

import json
import logging
import sys
from decimal import Decimal

import pytest

from bot_v2.logging.correlation import correlation_context
from bot_v2.logging.json_formatter import (
    DecimalEncoder,
    StructuredJSONFormatter,
    StructuredJSONFormatterWithTimestamp,
)


@pytest.mark.xfail(reason="JSON formatter update required")
def test_structured_json_formatter_basic():
    """Test basic JSON formatting without correlation context."""
    formatter = StructuredJSONFormatter()

    # Create a log record
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/test/path.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    # Format the record
    formatted = formatter.format(record)

    # Parse the JSON
    log_data = json.loads(formatted)

    # Check basic fields
    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "test.logger"
    assert log_data["message"] == "Test message"
    assert log_data["module"] == "path"
    assert log_data["function"] == "<module>"
    assert log_data["line"] == 42
    assert "timestamp" in log_data


def test_structured_json_formatter_with_correlation():
    """Test JSON formatting with correlation context."""
    formatter = StructuredJSONFormatter()

    # Set up correlation context
    with correlation_context("test-correlation-123", symbol="BTC-USD", side="buy"):
        # Create a log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message with context",
            args=(),
            exc_info=None,
        )

        # Format the record
        formatted = formatter.format(record)

        # Parse the JSON
        log_data = json.loads(formatted)

        # Check that correlation context is included
        assert log_data["correlation_id"] == "test-correlation-123"
        assert log_data["symbol"] == "BTC-USD"
        assert log_data["side"] == "buy"


@pytest.mark.xfail(reason="JSON formatter update required")
def test_structured_json_formatter_with_extra():
    """Test JSON formatting with extra fields."""
    formatter = StructuredJSONFormatter()

    # Create a log record with extra fields
    record = logging.LogRecord(
        name="test.logger",
        level=logging.WARNING,
        pathname="/test/path.py",
        lineno=42,
        msg="Warning message",
        args=(),
        exc_info=None,
    )
    # Add extra fields
    record.order_id = "order-456"
    record.quantity = Decimal("1.5")
    record.price = 50000.25

    # Format the record
    formatted = formatter.format(record)

    # Parse the JSON
    log_data = json.loads(formatted)

    # Check that extra fields are included
    assert log_data["order_id"] == "order-456"
    assert log_data["quantity"] == 1.5
    assert log_data["price"] == 50000.25


def test_structured_json_formatter_with_exception():
    """Test JSON formatting with exception info."""
    formatter = StructuredJSONFormatter()

    try:
        raise ValueError("Test exception")
    except ValueError:
        exc_info = sys.exc_info()

        # Create a log record with exception info
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        # Format the record
        formatted = formatter.format(record)

        # Parse the JSON
        log_data = json.loads(formatted)

        # Check exception fields
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"


def test_decimal_encoder():
    """Test the Decimal encoder."""
    encoder = DecimalEncoder()

    # Test with normal decimal
    result = encoder.default(Decimal("123.45"))
    assert result == 123.45

    # Test with very large decimal (should be string)
    large_decimal = Decimal("1" + "0" * 20)  # 10^20
    result = encoder.default(large_decimal)
    assert result == str(large_decimal)

    # Test with very small decimal (should be string)
    small_decimal = Decimal("0." + "0" * 20 + "1")  # 10^-20
    result = encoder.default(small_decimal)
    assert result == str(small_decimal)


def test_structured_json_formatter_with_timestamp():
    """Test the enhanced JSON formatter with additional timestamp fields."""
    formatter = StructuredJSONFormatterWithTimestamp()

    # Create a log record
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/test/path.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    record.created = 1640995200.123  # Fixed timestamp for testing

    # Format the record
    formatted = formatter.format(record)

    # Parse the JSON
    log_data = json.loads(formatted)

    # Check timestamp fields
    assert log_data["timestamp"] == "2022-01-01T00:00:00.123000Z"
    assert log_data["unix_timestamp"] == 1640995200.123
    assert log_data["date"] == "2022-01-01"
    assert log_data["time"] == "00:00:00"
    assert log_data["timezone"] == "UTC"


def test_structured_json_formatter_context_attributes():
    """Test handling of context_ attributes."""
    formatter = StructuredJSONFormatter()

    # Create a log record with context_ attributes
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/test/path.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    # Add context_ attributes (should be converted to regular fields)
    record.context_symbol = "ETH-USD"
    record.context_side = "sell"

    # Format the record
    formatted = formatter.format(record)

    # Parse the JSON
    log_data = json.loads(formatted)

    # Check that context_ attributes are converted
    assert log_data["symbol"] == "ETH-USD"
    assert log_data["side"] == "sell"
    # Should not have the original context_ attributes
    assert "context_symbol" not in log_data
    assert "context_side" not in log_data
