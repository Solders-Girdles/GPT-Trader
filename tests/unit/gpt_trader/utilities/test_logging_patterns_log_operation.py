from __future__ import annotations

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import (
    captured_logger,
    has_attr,
    messages,
)


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
