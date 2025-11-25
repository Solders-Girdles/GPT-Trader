from __future__ import annotations

import logging
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

from gpt_trader.utilities import logging_patterns as lp


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - logging API hook
        self.records.append(record)


@contextmanager
def captured_logger(name: str, level: int = logging.DEBUG) -> tuple[logging.Logger, ListHandler]:
    logger = logging.getLogger(name)
    handler = ListHandler()
    logger.handlers = [handler]
    logger.setLevel(level)
    logger.propagate = False
    try:
        yield logger, handler
    finally:
        logger.handlers = []


def messages(handler: ListHandler) -> list[str]:
    return [record.getMessage() for record in handler.records]


def has_attr(record: logging.LogRecord, attr: str, value: Any = None) -> bool:
    """Check if a log record has an attribute, optionally with a specific value."""
    if not hasattr(record, attr):
        return False
    if value is None:
        return True
    return getattr(record, attr) == value


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


def test_log_position_update_plain_logger_branch(caplog):
    caplog.set_level(logging.INFO, logger="position-branch")
    logger = logging.getLogger("position-branch")
    lp.log_position_update("SOL", position_size=Decimal("3"), logger=logger)

    assert any(
        rec.name == "position-branch" and has_attr(rec, "operation", "position_update")
        for rec in caplog.records
    )


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


def test_log_system_health_defaults_to_structured_logger():
    with captured_logger("health") as (_, handler):
        lp.log_system_health("healthy")

    record = handler.records[0]
    assert record.levelno == logging.INFO
    assert has_attr(record, "status", "healthy")


def test_log_system_health_plain_logger_branch(caplog):
    caplog.set_level(logging.WARNING, logger="health-branch")
    logger = logging.getLogger("health-branch")
    lp.log_system_health("degraded", component="risk", metrics={"error_rate": 0.2}, logger=logger)

    assert any(
        rec.name == "health-branch" and has_attr(rec, "operation", "health_check")
        for rec in caplog.records
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
    # Check that all records have component in extra (not in message)
    assert all(has_attr(rec, "component", "orders") for rec in handler.records)
    assert handler.records[0].getMessage().startswith("info message")


def test_get_logger_returns_structured_logger():
    slog = lp.get_logger("custom", component="risk")
    assert isinstance(slog, lp.StructuredLogger)
    with captured_logger("custom") as (_, handler):
        slog.info("hello", operation="greet")
    record = handler.records[0]
    assert has_attr(record, "operation", "greet")
