"""Tests for console logging utilities."""

from __future__ import annotations

import logging
import sys
from io import StringIO
from unittest.mock import Mock

import pytest

import gpt_trader.utilities.console_logging as console_logging_module
from gpt_trader.utilities.console_logging import ConsoleLogger


@pytest.fixture
def mock_output_stream() -> StringIO:
    return StringIO()


@pytest.fixture
def console_logger(mock_output_stream: StringIO) -> ConsoleLogger:
    return ConsoleLogger(enable_console=True, output_stream=mock_output_stream)


@pytest.fixture(autouse=True)
def reset_console_logger_singleton():
    console_logging_module._console_logger = None
    yield
    console_logging_module._console_logger = None


def test_init_default() -> None:
    logger = ConsoleLogger()

    assert logger.enable_console is True
    assert logger.output_stream == sys.stdout


def test_init_custom(mock_output_stream: StringIO) -> None:
    logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)

    assert logger.enable_console is False
    assert logger.output_stream == mock_output_stream


@pytest.mark.parametrize(
    ("method_name", "message", "kwargs", "expected"),
    [
        ("success", "Test success", {"test_id": "123"}, "✅ Test success"),
        ("error", "Test error", {"error_code": "E001"}, "❌ Test error"),
        ("warning", "Test warning", {"warning_type": "performance"}, "⚠️ Test warning"),
        ("info", "Test info", {"info_type": "general"}, "ℹ️ Test info"),
        ("data", "Data processed", {"records": 100}, "📊 Data processed"),
        ("trading", "Trade executed", {"symbol": "BTC-USD"}, "💰 Trade executed"),
        ("order", "Order placed", {"order_id": "123"}, "📝 Order placed"),
        ("position", "Position updated", {"symbol": "ETH-USD"}, "📈 Position updated"),
    ],
)
def test_logging_methods(
    console_logger: ConsoleLogger,
    mock_output_stream: StringIO,
    method_name: str,
    message: str,
    kwargs: dict[str, object],
    expected: str,
) -> None:
    getattr(console_logger, method_name)(message, **kwargs)

    output = mock_output_stream.getvalue()
    assert expected in output


def test_all_methods_when_console_disabled(mock_output_stream: StringIO) -> None:
    logger = ConsoleLogger(enable_console=False, output_stream=mock_output_stream)
    for method_name in [
        "success",
        "error",
        "warning",
        "info",
        "data",
        "trading",
        "order",
        "position",
        "cache",
        "storage",
        "network",
        "analysis",
        "ml",
    ]:
        getattr(logger, method_name)("message")

    assert mock_output_stream.getvalue() == ""


@pytest.mark.parametrize(
    ("enabled", "title", "char", "width", "expected"),
    [
        (True, "Test Section", "=", 30, "Test Section"),
        (True, "", "-", 20, "-" * 20),
        (False, "Hidden", "*", 10, ""),
    ],
)
def test_print_section_variants(
    mock_output_stream: StringIO,
    enabled: bool,
    title: str,
    char: str,
    width: int,
    expected: str,
) -> None:
    logger = ConsoleLogger(enable_console=enabled, output_stream=mock_output_stream)
    logger.print_section(title, char, width)

    output = mock_output_stream.getvalue()
    if enabled:
        assert expected in output
    else:
        assert output == ""


@pytest.mark.parametrize(
    ("enabled", "headers", "rows", "expected_substrings"),
    [
        (
            True,
            ["Symbol", "Price", "Quantity"],
            [
                ["BTC-USD", "50000", "1.0"],
                ["ETH-USD", "3000", "10.0"],
                ["SOL-USD", "150", "20.0", "ignored"],
            ],
            ["Symbol", "Price", "Quantity", "BTC-USD", "ETH-USD", "-"],
        ),
        (True, ["Symbol", "Price"], [], []),
        (False, ["Symbol", "Price"], [["BTC-USD", "50000"]], []),
    ],
)
def test_print_table_variants(
    mock_output_stream: StringIO,
    enabled: bool,
    headers: list[str],
    rows: list[list[str]],
    expected_substrings: list[str],
) -> None:
    logger = ConsoleLogger(enable_console=enabled, output_stream=mock_output_stream)
    logger.print_table(headers, rows)

    output = mock_output_stream.getvalue()
    if expected_substrings:
        for text in expected_substrings:
            assert text in output
    else:
        assert output == ""


@pytest.mark.parametrize(
    ("enabled", "key", "value", "indent", "expected"),
    [
        (True, "Test Key", "Test Value", 0, "Test Key: Test Value"),
        (True, "Nested Key", "Nested Value", 2, "      Nested Key: Nested Value"),
        (False, "Test Key", "Test Value", 0, ""),
    ],
)
def test_print_key_value_variants(
    mock_output_stream: StringIO,
    enabled: bool,
    key: str,
    value: str,
    indent: int,
    expected: str,
) -> None:
    logger = ConsoleLogger(enable_console=enabled, output_stream=mock_output_stream)
    logger.printKeyValue(key, value, indent=indent)

    output = mock_output_stream.getvalue()
    if enabled:
        assert expected in output
    else:
        assert output == ""


def test_structured_logging_integration(console_logger: ConsoleLogger, caplog) -> None:
    caplog.set_level(logging.INFO, logger="gpt_trader.utilities.console_logging")

    console_logger.success("Test message", test_param="value")

    assert any(
        record.name == "gpt_trader.utilities.console_logging"
        and "Test message" in record.getMessage()
        for record in caplog.records
    )


def test_output_stream_error_handling(mock_output_stream: StringIO) -> None:
    mock_output_stream.write = Mock(side_effect=Exception("Stream error"))

    logger = ConsoleLogger(enable_console=True, output_stream=mock_output_stream)
    logger.success("Test message")

    mock_output_stream.write.assert_called()
