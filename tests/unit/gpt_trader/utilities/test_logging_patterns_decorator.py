from __future__ import annotations

from contextlib import contextmanager

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import captured_logger


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
