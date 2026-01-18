from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import (
    captured_logger,
    has_attr,
)


def test_log_market_data_update_supports_optional_fields():
    with captured_logger("market_data") as (_, handler):
        lp.log_market_data_update(
            "BTC",
            price=Decimal("42000"),
            volume=Decimal("12.5"),
            timestamp=123456.0,
        )

    record = handler.records[0]
    assert has_attr(record, "operation", "market_data_update")
    assert has_attr(record, "symbol", "BTC")
    assert has_attr(record, "price", Decimal("42000"))


def test_log_market_data_update_without_optional_fields():
    with captured_logger("market_data") as (_, handler):
        lp.log_market_data_update("DOGE", price=Decimal("0.01"))

    record = handler.records[0]
    assert has_attr(record, "operation", "market_data_update")
    assert has_attr(record, "symbol", "DOGE")
    assert not has_attr(record, "volume")
    assert not has_attr(record, "timestamp")


def test_log_market_data_update_with_plain_logger_and_optional_fields():
    with captured_logger("market_plain") as (logger, handler):
        lp.log_market_data_update(
            "ETH",
            price=Decimal("1234.56"),
            volume=Decimal("99.9"),
            timestamp=987654.0,
            logger=logger,
        )

    record = handler.records[0]
    assert record.name == "market_plain"
    assert has_attr(record, "symbol", "ETH")
    assert has_attr(record, "price", Decimal("1234.56"))


def test_log_market_data_update_plain_logger_branch(caplog):
    caplog.set_level(logging.DEBUG, logger="market-branch")
    logger = logging.getLogger("market-branch")
    lp.log_market_data_update(
        "XRP", price=Decimal("0.5"), volume=Decimal("10"), timestamp=1_234.0, logger=logger
    )

    assert any(
        rec.name == "market-branch" and has_attr(rec, "operation", "market_data_update")
        for rec in caplog.records
    )


def test_log_market_data_update_wraps_logging_logger(monkeypatch):
    seen: dict[str, Any] = {}
    original = lp.StructuredLogger

    def spy_structured(name: str, component: str | None = None):
        seen["name"] = name
        return original(name, component=component)

    monkeypatch.setattr(lp, "StructuredLogger", spy_structured)

    with captured_logger("market-wrap") as (logger, handler):
        lp.log_market_data_update("SOL", price=Decimal("19.5"), logger=logger)

    assert seen["name"] == "market-wrap"
    assert handler.records[0].name == "market-wrap"
