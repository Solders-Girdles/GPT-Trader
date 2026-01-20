"""Tests for core logging patterns: StructuredLogger, log_operation, log_execution."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import (
    captured_logger,
    has_attr,
    messages,
)

# ============================================================================
# StructuredLogger Tests
# ============================================================================


def test_structured_logger_methods_apply_component():
    with captured_logger("structured") as (_, handler):
        slog = lp.StructuredLogger("structured", component="orders")
        slog.info("info message", operation="test")
        slog.warning("warn message", operation="test")
        slog.error("error message", operation="test")
        slog.debug("debug message", operation="test")
        slog.critical("critical message", operation="test")
        slog.log(logging.INFO, "generic message", operation="test")

    assert len(handler.records) == 6
    assert all(has_attr(rec, "component", "orders") for rec in handler.records)
    assert handler.records[0].getMessage().startswith("info message")


def test_get_logger_returns_structured_logger():
    slog = lp.get_logger("custom", component="risk")
    assert isinstance(slog, lp.StructuredLogger)
    with captured_logger("custom") as (_, handler):
        slog.info("hello", operation="greet")
    record = handler.records[0]
    assert has_attr(record, "operation", "greet")


# ============================================================================
# log_operation Tests
# ============================================================================


def test_log_operation_with_plain_logger():
    with captured_logger("operation") as (logger, handler):
        with lp.log_operation("fetch", logger=logger, symbol="BTC"):
            pass

    start_msg, end_msg = messages(handler)
    start_rec, end_rec = handler.records

    assert "Started fetch" in start_msg
    assert has_attr(start_rec, "operation", "fetch")
    assert has_attr(start_rec, "symbol", "BTC")
    assert "Completed fetch" in end_msg
    assert has_attr(end_rec, "duration_ms")
    assert handler.records[0].name == "operation"


def test_log_operation_with_default_logger():
    with captured_logger("operation") as (_, handler):
        with lp.log_operation("sync"):
            pass

    start, end = messages(handler)
    assert "Started sync" in start
    assert "Completed sync" in end


# ============================================================================
# log_execution Decorator Tests
# ============================================================================


def test_log_execution_includes_args_and_result():
    with captured_logger("decorator") as (logger, handler):

        @lp.log_execution(
            operation="compute",
            logger=logger,
            include_args=True,
            include_result=True,
        )
        def add(a: int, b: int) -> int:
            return a + b

        assert add(1, 2) == 3

    assert any("Started compute" in rec.getMessage() for rec in handler.records)
    assert any("Completed compute" in rec.getMessage() for rec in handler.records)
    assert any("Result: 3" in rec.getMessage() for rec in handler.records)


def test_log_execution_uses_module_logger_when_none():
    module_logger_name = __name__
    with captured_logger(module_logger_name) as (_, handler):

        @lp.log_execution(include_result=True)
        def multiply(a: int, b: int) -> int:
            return a * b

        assert multiply(2, 4) == 8

    assert any("Started multiply" in record.getMessage() for record in handler.records)
    assert any("Result: 8" in record.getMessage() for record in handler.records)


def test_log_execution_with_structured_logger():
    with captured_logger("decorator_struct") as (_, handler):
        slog = lp.StructuredLogger("decorator_struct", component="exec")

        @lp.log_execution(logger=slog, include_args=True, include_result=True)
        def subtract(a: int, b: int) -> int:
            return a - b

        assert subtract(5, 2) == 3

    assert any("Started subtract" in rec.getMessage() for rec in handler.records)
    assert any("Result: 3" in rec.getMessage() for rec in handler.records)


def test_log_execution_excludes_callable_args(monkeypatch):
    captured_context: dict[str, object] = {}

    @contextmanager
    def fake_log_operation(operation, logger, level, **context):
        captured_context["operation"] = operation
        captured_context["context"] = dict(context)
        yield

    monkeypatch.setattr(lp, "log_operation", fake_log_operation)

    @lp.log_execution(include_args=True)
    def with_callback(value: int, cb):
        cb(value)
        return value

    marker: list[int] = []
    assert with_callback(7, lambda x: marker.append(x)) == 7
    assert marker == [7]

    ctx = captured_context["context"]
    assert ctx["arg_0"] == "7"
    assert "arg_1" not in ctx


def test_log_execution_no_result_logging_when_none():
    with captured_logger("decorator_none") as (logger, handler):

        @lp.log_execution(logger=logger, include_result=True)
        def do_nothing():
            return None

        assert do_nothing() is None

    assert not any("Result:" in rec.getMessage() for rec in handler.records)
