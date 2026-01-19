from __future__ import annotations

import logging

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import (
    captured_logger,
    has_attr,
)


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
