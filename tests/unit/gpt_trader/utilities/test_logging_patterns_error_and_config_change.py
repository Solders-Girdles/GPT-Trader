from __future__ import annotations

import logging
from typing import Any

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import (
    captured_logger,
    has_attr,
)


def test_log_error_with_context():
    with captured_logger("error") as (_, handler):
        try:
            raise ValueError("boom")
        except ValueError as exc:
            lp.log_error_with_context(exc, "sync", component="orders", order_id="xyz")

    record = handler.records[0]
    assert has_attr(record, "operation", "sync")
    assert has_attr(record, "error_type", "ValueError")
    assert has_attr(record, "order_id", "xyz")
    assert has_attr(record, "component", "orders")


def test_log_error_with_context_without_component():
    with captured_logger("error") as (_, handler):
        try:
            raise RuntimeError("kaboom")
        except RuntimeError as exc:
            lp.log_error_with_context(exc, "sync")

    record = handler.records[0]
    assert has_attr(record, "operation", "sync")
    assert not has_attr(record, "component")


def test_log_configuration_change_includes_component():
    with captured_logger("config") as (_, handler):
        lp.log_configuration_change("api_url", "old", "new", component="service")

    record = handler.records[0]
    assert has_attr(record, "operation", "config_change")
    assert has_attr(record, "component", "service")


def test_log_configuration_change_without_component():
    with captured_logger("config") as (_, handler):
        lp.log_configuration_change("mode", "old", "new")

    record = handler.records[0]
    assert has_attr(record, "operation", "config_change")
    assert not has_attr(record, "component")


def test_log_configuration_change_with_plain_logger():
    with captured_logger("config_plain") as (logger, handler):
        lp.log_configuration_change("timeout", None, 30, logger=logger)

    record = handler.records[0]
    assert record.name == "config_plain"
    assert has_attr(record, "operation", "config_change")


def test_log_configuration_change_wraps_logging_logger(monkeypatch):
    seen: dict[str, Any] = {}
    original = lp.StructuredLogger

    def spy_structured(name: str, component: str | None = None):
        seen["name"] = name
        return original(name, component=component)

    monkeypatch.setattr(lp, "StructuredLogger", spy_structured)

    with captured_logger("config-wrap") as (logger, handler):
        lp.log_configuration_change("mode", "old", "new", logger=logger)

    assert seen["name"] == "config-wrap"
    assert handler.records[0].name == "config-wrap"


def test_log_configuration_change_plain_logger_branch(caplog):
    caplog.set_level(logging.INFO, logger="config-branch")
    logger = logging.getLogger("config-branch")
    lp.log_configuration_change("api_mode", "old", "new", logger=logger)

    assert any(
        rec.name == "config-branch" and has_attr(rec, "operation", "config_change")
        for rec in caplog.records
    )
