"""Tests for console logging global helpers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

import gpt_trader.utilities.console_logging as console_logging_module
from gpt_trader.utilities.console_logging import (
    ConsoleLogger,
    console_analysis,
    console_cache,
    console_data,
    console_error,
    console_info,
    console_key_value,
    console_ml,
    console_network,
    console_order,
    console_position,
    console_section,
    console_storage,
    console_success,
    console_table,
    console_trading,
    console_warning,
    get_console_logger,
)


@pytest.fixture
def console_logger_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_logger = Mock()
    monkeypatch.setattr(console_logging_module, "ConsoleLogger", Mock(return_value=mock_logger))
    return mock_logger


@pytest.fixture
def get_console_logger_mock(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_logger = Mock()
    monkeypatch.setattr(
        console_logging_module, "get_console_logger", Mock(return_value=mock_logger)
    )
    return mock_logger


@pytest.fixture(autouse=True)
def reset_console_logger_singleton():
    console_logging_module._console_logger = None
    yield
    console_logging_module._console_logger = None


def test_end_to_end_logging_workflow(console_logger_mock: Mock) -> None:
    """Test complete logging workflow."""
    console_section("Trading Session")
    console_success("Connected to broker")
    console_order("Order placed", symbol="BTC-USD", quantity=1.0)
    console_position("Position updated", symbol="BTC-USD", quantity=1.0)
    console_data("Market data received", symbols=["BTC-USD", "ETH-USD"])

    console_logging_module.ConsoleLogger.assert_called_once_with(enable_console=True)
    assert console_logging_module._console_logger is console_logger_mock

    console_logger_mock.print_section.assert_called_once_with("Trading Session", "=", 50)
    console_logger_mock.success.assert_called_once_with("Connected to broker")
    console_logger_mock.order.assert_called_once_with(
        "Order placed", symbol="BTC-USD", quantity=1.0
    )
    console_logger_mock.position.assert_called_once_with(
        "Position updated", symbol="BTC-USD", quantity=1.0
    )
    console_logger_mock.data.assert_called_once_with(
        "Market data received", symbols=["BTC-USD", "ETH-USD"]
    )


def test_table_formatting_complex_data(console_logger_mock: Mock) -> None:
    """Test table formatting with complex data."""
    headers = ["Symbol", "Price", "Quantity", "Value", "P&L", "P&L %"]
    rows = [
        ["BTC-USD", "50,123.45", "1.5", "75,185.18", "+1,234.56", "+1.67%"],
        ["ETH-USD", "3,456.78", "10.2", "35,259.16", "-123.45", "-0.35%"],
        ["SOL-USD", "123.45", "100.0", "12,345.00", "+567.89", "+4.81%"],
    ]

    console_table(headers, rows)
    console_table(headers, [])
    single_row = [rows[0]]
    console_table(headers, single_row)

    assert console_logger_mock.print_table.call_count == 3
    assert console_logger_mock.print_table.call_args_list[0].args == (headers, rows)
    assert console_logger_mock.print_table.call_args_list[1].args == (headers, [])
    assert console_logger_mock.print_table.call_args_list[2].args == (headers, single_row)


def test_error_handling_in_global_functions() -> None:
    try:
        console_success("Test message")
        console_error("Test error")
        console_warning("Test warning")
        console_info("Test info")
    except Exception as exc:
        pytest.fail(f"Global console functions raised exception: {exc}")


def test_function_parameter_passing(get_console_logger_mock: Mock) -> None:
    """Test that parameters are correctly passed to underlying logger."""
    console_success("Test", string_param="value", int_param=42, float_param=3.14)
    get_console_logger_mock.success.assert_called_once_with(
        "Test", string_param="value", int_param=42, float_param=3.14
    )

    get_console_logger_mock.reset_mock()
    console_error("Error", exception=ValueError("test"), none_param=None)
    get_console_logger_mock.error.assert_called_once()
    call_args = get_console_logger_mock.error.call_args
    assert call_args[0][0] == "Error"
    assert "exception" in call_args[1]
    assert "none_param" in call_args[1]
    assert call_args[1]["none_param"] is None


def test_multiple_function_calls(get_console_logger_mock: Mock) -> None:
    console_success("Success 1")
    console_error("Error 1")
    console_warning("Warning 1")
    console_info("Info 1")

    assert get_console_logger_mock.success.call_count == 1
    assert get_console_logger_mock.error.call_count == 1
    assert get_console_logger_mock.warning.call_count == 1
    assert get_console_logger_mock.info.call_count == 1

    expected_calls = [
        (("Success 1",), {}),
        (("Error 1",), {}),
        (("Warning 1",), {}),
        (("Info 1",), {}),
    ]

    actual_calls = []
    for method_name in ["success", "error", "warning", "info"]:
        method = getattr(get_console_logger_mock, method_name)
        if method.called:
            actual_calls.append((method.call_args[0], method.call_args[1]))

    assert actual_calls == expected_calls


@pytest.mark.parametrize(
    ("wrapper", "method_name", "args", "kwargs"),
    [
        (console_success, "success", ("Test success",), {"test_id": "123"}),
        (console_error, "error", ("Test error",), {"error_code": "E001"}),
        (console_warning, "warning", ("Test warning",), {"warning_type": "performance"}),
        (console_info, "info", ("Test info",), {"info_type": "general"}),
        (console_data, "data", ("Test data",), {"records": 100}),
        (console_trading, "trading", ("Test trading",), {"symbol": "BTC-USD"}),
        (console_order, "order", ("Test order",), {"order_id": "123"}),
        (console_position, "position", ("Test position",), {"symbol": "ETH-USD"}),
        (console_cache, "cache", ("Test cache",), {"cache_key": "test_key"}),
        (console_storage, "storage", ("Test storage",), {"file": "test.json"}),
        (console_network, "network", ("Test network",), {"endpoint": "api.test.com"}),
        (console_analysis, "analysis", ("Test analysis",), {"metric": "sharpe"}),
        (console_ml, "ml", ("Test ML",), {"model": "test_model"}),
        (console_section, "print_section", ("Test Section", "=", 40), {}),
        (console_table, "print_table", (["Symbol", "Price"], [["BTC-USD", "50000"]]), {}),
        (console_key_value, "printKeyValue", ("Test Key", "Test Value", 2), {}),
    ],
)
def test_console_wrapper_functions(
    get_console_logger_mock: Mock,
    wrapper,
    method_name: str,
    args: tuple,
    kwargs: dict[str, object],
) -> None:
    wrapper(*args, **kwargs)
    getattr(get_console_logger_mock, method_name).assert_called_once_with(*args, **kwargs)


def _capture_fallback_print(monkeypatch: pytest.MonkeyPatch, failing_stream: object) -> list[str]:
    recorded: list[str] = []

    def fake_print(*args, **kwargs):
        if kwargs.get("file") is failing_stream:
            raise Exception("stream failure")
        recorded.append(args[0])

    monkeypatch.setattr("builtins.print", fake_print)
    return recorded


def test_print_section_fallback_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_stream = object()
    recorded = _capture_fallback_print(monkeypatch, failing_stream)
    logger = ConsoleLogger(enable_console=True, output_stream=failing_stream)

    logger.print_section("Fallback", "#", 10)

    assert recorded
    assert "Fallback" in recorded[0]


def test_print_table_fallback_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_stream = object()
    recorded = _capture_fallback_print(monkeypatch, failing_stream)
    logger = ConsoleLogger(enable_console=True, output_stream=failing_stream)

    headers = ["Name"]
    rows = [["Alice"], ["Bob"]]
    logger.print_table(headers, rows)

    assert recorded[0].strip() == "Name"
    assert recorded[1].strip().startswith("-")
    assert recorded[2].strip().startswith("Alice")
    assert recorded[3].strip().startswith("Bob")


def test_print_key_value_fallback_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failing_stream = object()
    recorded = _capture_fallback_print(monkeypatch, failing_stream)
    logger = ConsoleLogger(enable_console=True, output_stream=failing_stream)

    logger.printKeyValue("Key", "Value", indent=1)

    assert recorded == ["   Key: Value"]


def test_get_console_logger_singleton_behavior() -> None:
    logger = get_console_logger(enable_console=False)
    assert logger.enable_console is False
    assert get_console_logger() is logger

    logger.enable_console = True
    assert get_console_logger(enable_console=False) is logger
    assert logger.enable_console is True
