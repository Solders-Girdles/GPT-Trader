"""Tests for domain and system logging helpers."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from gpt_trader.utilities import logging_patterns as lp
from tests.unit.gpt_trader.utilities.logging_patterns_test_helpers import (
    captured_logger,
    has_attr,
)


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


def test_log_system_health_warns_on_non_healthy():
    with captured_logger("health") as (logger, handler):
        lp.log_system_health(
            "degraded",
            component="monitor",
            metrics={"error_rate": 0.5},
            logger=logger,
        )

    record = handler.records[0]
    assert record.levelno == logging.WARNING
    assert has_attr(record, "operation", "health_check")
    assert has_attr(record, "component", "monitor")


def test_log_system_health_wraps_logging_logger(monkeypatch):
    seen: dict[str, Any] = {}
    original = lp.StructuredLogger

    class SpyStructured(original):
        def __init__(self, name: str, component: str | None = None) -> None:
            super().__init__(name, component=component)
            seen["name"] = name

        def log(self, level: int, message: str, **context: Any) -> None:  # type: ignore[override]
            seen["level"] = level
            seen["context"] = context
            super().log(level, message, **context)

    monkeypatch.setattr(lp, "StructuredLogger", SpyStructured)

    with captured_logger("health-wrap") as (logger, handler):
        lp.log_system_health(
            "degraded",
            component="execution",
            metrics={"latency_ms": 120},
            logger=logger,
        )

    assert seen["name"] == "health-wrap"
    assert seen["level"] == logging.WARNING
    assert seen["context"]["latency_ms"] == 120
    assert handler.records[0].name == "health-wrap"


def test_log_system_health_defaults_to_structured_logger():
    with captured_logger("health") as (_, handler):
        lp.log_system_health("healthy")

    record = handler.records[0]
    assert record.levelno == logging.INFO
    assert has_attr(record, "status", "healthy")


class _Level:
    """Stand-in for monitoring.system.LogLevel (``.value`` is the level name)."""

    def __init__(self, value: str) -> None:
        self.value = value


def test_log_event_emits_structured_record_with_fields():
    with captured_logger("monitoring") as (_, handler):
        slog = lp.StructuredLogger("monitoring")
        slog.log_event(
            level=_Level("INFO"),
            event_type="order_preview",
            message="Order preview generated",
            component="TradingEngine",
            symbol="BTC-USD",
            side="BUY",
        )

    record = handler.records[0]
    assert record.levelno == logging.INFO
    assert record.getMessage() == "Order preview generated"
    assert has_attr(record, "event_type", "order_preview")
    assert has_attr(record, "component", "TradingEngine")
    assert has_attr(record, "symbol", "BTC-USD")
    assert has_attr(record, "side", "BUY")


def test_log_event_coerces_string_and_int_levels():
    with captured_logger("monitoring") as (_, handler):
        slog = lp.StructuredLogger("monitoring")
        slog.log_event(level="WARNING", event_type="warn", message="warn")
        slog.log_event(level=logging.ERROR, event_type="err", message="err")
        slog.log_event(event_type="default", message="default")

    assert handler.records[0].levelno == logging.WARNING
    assert handler.records[1].levelno == logging.ERROR
    assert handler.records[2].levelno == logging.INFO


def test_log_order_submission_records_event():
    with captured_logger("monitoring") as (_, handler):
        slog = lp.StructuredLogger("monitoring")
        slog.log_order_submission(
            client_order_id="client-123",
            symbol="BTC-USD",
            side="BUY",
            order_type="LIMIT",
            quantity=1.5,
            price=50000.0,
        )

    record = handler.records[0]
    assert record.levelno == logging.INFO
    assert has_attr(record, "event_type", "order_submission")
    assert has_attr(record, "operation", "order_submit")
    assert has_attr(record, "client_order_id", "client-123")
    assert has_attr(record, "symbol", "BTC-USD")
    assert has_attr(record, "side", "BUY")
    assert has_attr(record, "order_type", "LIMIT")
    assert has_attr(record, "quantity", 1.5)
    assert has_attr(record, "price", 50000.0)


def test_log_order_submission_accepts_none_price():
    with captured_logger("monitoring") as (_, handler):
        slog = lp.StructuredLogger("monitoring")
        slog.log_order_submission(
            client_order_id="client-123",
            symbol="BTC-USD",
            side="BUY",
            order_type="MARKET",
            quantity=1.0,
            price=None,
        )

    record = handler.records[0]
    assert has_attr(record, "price")
    assert record.price is None  # type: ignore[attr-defined]


def test_log_order_status_change_records_event():
    with captured_logger("monitoring") as (_, handler):
        slog = lp.StructuredLogger("monitoring")
        slog.log_order_status_change(
            order_id="order-123",
            client_order_id="client-123",
            from_status=None,
            to_status="open",
        )

    record = handler.records[0]
    assert record.levelno == logging.INFO
    assert has_attr(record, "event_type", "order_status_change")
    assert has_attr(record, "operation", "order_status")
    assert has_attr(record, "order_id", "order-123")
    assert has_attr(record, "client_order_id", "client-123")
    assert has_attr(record, "to_status", "open")


def test_log_order_status_change_includes_optional_reason():
    with captured_logger("monitoring") as (_, handler):
        slog = lp.StructuredLogger("monitoring")
        slog.log_order_status_change(
            order_id="",
            client_order_id="",
            from_status=None,
            to_status="REJECTED",
            reason="paused",
        )

    record = handler.records[0]
    assert has_attr(record, "to_status", "REJECTED")
    assert has_attr(record, "reason", "paused")
