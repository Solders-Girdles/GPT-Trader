"""Tests for domain-specific logging: trades, positions, market data."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import (
    captured_logger,
    has_attr,
)

# ============================================================================
# Trade Event Tests
# ============================================================================


def test_log_trade_event_includes_all_context():
    with captured_logger("trade") as (logger, handler):
        lp.log_trade_event(
            "order_filled",
            "BTC-PERP",
            side="buy",
            quantity=Decimal("0.1"),
            price=Decimal("42000"),
            order_id="abc",
            logger=logger,
        )

    record = handler.records[0]
    assert has_attr(record, "operation", "trade_event")
    assert has_attr(record, "symbol", "BTC-PERP")
    assert has_attr(record, "side", "buy")
    assert has_attr(record, "quantity", Decimal("0.1"))
    assert has_attr(record, "price", Decimal("42000"))
    assert has_attr(record, "order_id", "abc")


def test_log_trade_event_defaults_to_structured_logger():
    with captured_logger("trading") as (_, handler):
        lp.log_trade_event("order_cancelled", "ETH-PERP")

    record = handler.records[0]
    assert has_attr(record, "operation", "trade_event")
    assert has_attr(record, "symbol", "ETH-PERP")


def test_log_trade_event_with_structured_logger():
    with captured_logger("trade_structured") as (_, handler):
        structured = lp.StructuredLogger("trade_structured", component="desk")
        lp.log_trade_event("fill", "SOL", logger=structured)

    record = handler.records[0]
    assert has_attr(record, "operation", "trade_event")
    assert has_attr(record, "symbol", "SOL")
    assert has_attr(record, "component", "desk")


# ============================================================================
# Position Update Tests
# ============================================================================


def test_log_position_update_adds_metrics():
    with captured_logger("position") as (logger, handler):
        lp.log_position_update(
            "ETH",
            position_size=Decimal("2"),
            unrealized_pnl=Decimal("5.5"),
            equity=Decimal("100"),
            leverage=3.5,
            logger=logger,
        )

    record = handler.records[0]
    assert has_attr(record, "operation", "position_update")
    assert has_attr(record, "position_size", Decimal("2"))
    assert has_attr(record, "unrealized_pnl", Decimal("5.5"))
    assert has_attr(record, "equity", Decimal("100"))
    assert has_attr(record, "leverage", 3.5)


def test_log_position_update_defaults_to_structured_logger():
    with captured_logger("position") as (_, handler):
        lp.log_position_update("BTC", position_size=Decimal("1"))

    record = handler.records[0]
    assert has_attr(record, "operation", "position_update")


def test_log_position_update_with_plain_logger():
    with captured_logger("plain-position") as (logger, handler):
        lp.log_position_update("DOGE", position_size=Decimal("5"), logger=logger)

    record = handler.records[0]
    assert record.name == "plain-position"
    assert has_attr(record, "operation", "position_update")


def test_log_position_update_wraps_logging_logger(monkeypatch):
    seen: dict[str, Any] = {}
    original = lp.StructuredLogger

    def spy_structured(name: str, component: str | None = None):
        seen["name"] = name
        return original(name, component=component)

    monkeypatch.setattr(lp, "StructuredLogger", spy_structured)

    with captured_logger("position-wrap") as (logger, handler):
        lp.log_position_update("BTC", position_size=Decimal("1"), logger=logger)

    assert seen["name"] == "position-wrap"
    assert handler.records[0].name == "position-wrap"


# ============================================================================
# Market Data Tests
# ============================================================================


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


def test_log_market_data_update_with_plain_logger():
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
