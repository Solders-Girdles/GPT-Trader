from __future__ import annotations

import logging
from decimal import Decimal

import pytest

from bot_v2.utilities import logging_patterns


def test_structured_logger_formats_context(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    structured = logging_patterns.StructuredLogger("tests.logging", component="matcher")

    caplog.set_level(logging.INFO, "tests.logging")
    structured.info(
        "Order executed",
        symbol="BTC-USD",
        price=Decimal("123.45000000"),
        quantity=Decimal("1.20000000"),
        ignored_field="skip-me",
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("Order executed | component=matcher")
    assert "symbol=BTC-USD" in message
    assert "price=123.45" in message
    assert "quantity=1.2" in message
    assert "ignored_field" not in message


def test_log_operation_records_duration(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    caplog.set_level(logging.INFO, "operation")
    with logging_patterns.log_operation("refresh", symbol="ETH-USD"):
        pass

    assert len(caplog.records) == 2
    start_msg = caplog.records[0].message
    end_msg = caplog.records[1].message
    assert start_msg == "Started refresh | operation=refresh symbol=ETH-USD"
    assert end_msg.startswith("Completed refresh |")
    duration_token = next(part for part in end_msg.split() if part.startswith("duration_ms="))
    assert float(duration_token.split("=")[1]) >= 0
    assert "symbol=ETH-USD" in end_msg


def test_log_trade_event_includes_context(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    caplog.set_level(logging.INFO, "trading")

    logging_patterns.log_trade_event(
        "order_filled",
        "BTC-USD",
        side="buy",
        quantity=Decimal("1.5"),
        price=Decimal("25000"),
        order_id="abc123",
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "Trade event: order_filled" in message
    assert "operation=trade_event" in message
    assert "symbol=BTC-USD" in message
    assert "side=buy" in message
    assert "order_id=abc123" in message


def test_log_position_update_uses_structured_logger(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    structured = logging_patterns.StructuredLogger("position", component="risk")

    caplog.set_level(logging.INFO, "position")
    logging_patterns.log_position_update(
        "ETH-USD",
        Decimal("2.0"),
        unrealized_pnl=Decimal("123.45"),
        equity=Decimal("5000"),
        leverage=2.5,
        logger=structured,
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("Position update: ETH-USD | component=risk")
    assert "position_size=2" in message or "position_size=2.0" in message
    assert "pnl=123.45" in message
    assert "equity=5000" in message
    assert "leverage=2.5" in message


def test_log_error_with_context(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    caplog.set_level(logging.ERROR, "error")

    logging_patterns.log_error_with_context(
        ValueError("invalid input"),
        "sync_orders",
        component="engine",
        user="alice",
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("Error in sync_orders: invalid input | component=engine")
    assert "error_type=ValueError" in message
    assert "user=alice" not in message  # context key not in LOG_FIELDS


def test_log_configuration_change_with_standard_logger(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.clear()
    base_logger = logging.getLogger("config-change")
    caplog.set_level(logging.INFO, "config-change")

    captured_kwargs: dict[str, str] = {}
    original_format = logging_patterns.StructuredLogger._format_message

    def spy_format(self, message: str, **kwargs: object) -> str:  # type: ignore[override]
        captured_kwargs.update(kwargs)
        return original_format(self, message, **kwargs)

    monkeypatch.setattr(logging_patterns.StructuredLogger, "_format_message", spy_format)

    logging_patterns.log_configuration_change(
        "risk_limit",
        None,
        Decimal("5000"),
        component="risk",
        logger=base_logger,
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("Configuration changed: risk_limit |")
    assert "operation=config_change" in message
    assert "component=risk" in message
    assert captured_kwargs["config_key"] == "risk_limit"
    assert captured_kwargs["old_value"] == "None"
    assert captured_kwargs["new_value"] == "5000"


def test_log_market_data_update_accepts_plain_logger(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    caplog.clear()
    base_logger = logging.getLogger("market.custom")
    caplog.set_level(logging.DEBUG, "market.custom")

    captured_kwargs: dict[str, object] = {}
    original_format = logging_patterns.StructuredLogger._format_message

    def spy_format(self, message: str, **kwargs: object) -> str:  # type: ignore[override]
        captured_kwargs.update(kwargs)
        return original_format(self, message, **kwargs)

    monkeypatch.setattr(logging_patterns.StructuredLogger, "_format_message", spy_format)

    logging_patterns.log_market_data_update(
        "BTC-USD",
        Decimal("30000"),
        volume=Decimal("12.5"),
        timestamp=1700000000.0,
        logger=base_logger,
    )

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert message.startswith("Market data update: BTC-USD |")
    assert "price=30000" in message
    assert "operation=market_data_update" in message
    assert captured_kwargs["volume"] == Decimal("12.5")
    assert captured_kwargs["timestamp"] == 1700000000.0


def test_log_system_health_levels(caplog: pytest.LogCaptureFixture) -> None:
    caplog.clear()
    caplog.set_level(logging.INFO, "health")

    logging_patterns.log_system_health(
        "healthy",
        component="match-engine",
        metrics={"pnl": Decimal("10.00")},
    )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.INFO
    assert "Health status: healthy" in record.message
    assert "component=match-engine" in record.message
    assert "pnl=10" in record.message

    caplog.clear()
    logging_patterns.log_system_health(
        "degraded",
        component="match-engine",
        metrics={"pnl": Decimal("-1.5")},
    )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "Health status: degraded" in record.message
    assert "pnl=-1.5" in record.message


def test_get_logger_returns_structured_logger() -> None:
    structured = logging_patterns.get_logger("custom", component="broker")
    assert isinstance(structured, logging_patterns.StructuredLogger)
    assert structured.component == "broker"
