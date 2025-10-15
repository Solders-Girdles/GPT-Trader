from __future__ import annotations

import logging

import pytest

from bot_v2.monitoring.system.logger import LogLevel
from bot_v2.utilities.telemetry import emit_metric


class DummyStore:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def append_metric(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class PositionalOnlyStore(DummyStore):
    def append_metric(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if kwargs:
            raise TypeError("append_metric takes positional args only")


class FailingStore(DummyStore):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def append_metric(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise self._exc


class LoggerSpy:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def log_event(self, level: LogLevel, event_type: str, message: str, **kwargs):
        self.events.append((level, event_type, message, kwargs))


def test_emit_metric_ignores_missing_event_store() -> None:
    # Should be a no-op when event store is None
    emit_metric(None, "bot-123", {"event_type": "heartbeat"})


def test_emit_metric_normalizes_event_type_fields() -> None:
    store = DummyStore()

    emit_metric(store, "bot-123", {"event_type": "ws_mark_update"})

    assert len(store.calls) == 1
    args, kwargs = store.calls[0]
    assert args == ()
    assert kwargs["bot_id"] == "bot-123"
    metrics = kwargs["metrics"]
    assert metrics["event_type"] == "ws_mark_update"
    assert metrics["type"] == "ws_mark_update"


def test_emit_metric_defaults_missing_event_type() -> None:
    store = DummyStore()

    emit_metric(store, "bot-456", {"symbol": "ETH-USD"})

    metrics = store.calls[0][1]["metrics"]
    assert metrics["event_type"] == "metric"
    assert metrics["type"] == "metric"


def test_emit_metric_retries_with_positional_arguments() -> None:
    store = PositionalOnlyStore()

    emit_metric(store, "bot-pos", {"event_type": "latency_probe"})

    assert len(store.calls) == 2
    # First attempt used kwargs and failed
    assert store.calls[0][1] != {}
    # Second attempt succeeded with positional args
    assert store.calls[1][0] == (
        "bot-pos",
        {"event_type": "latency_probe", "type": "latency_probe"},
    )
    assert store.calls[1][1] == {}


def test_emit_metric_logs_using_production_logger_on_failure() -> None:
    error = RuntimeError("event-store-down")
    store = FailingStore(error)
    logger_spy = LoggerSpy()

    emit_metric(store, "bot-prod", {"event_type": "account_snapshot"}, logger=logger_spy)

    assert len(logger_spy.events) == 1
    level, event_type, message, payload = logger_spy.events[0]
    assert level is LogLevel.DEBUG
    assert event_type == "metric_emit_failed"
    assert message == "Failed to emit metric to event store"
    assert payload["error"] == str(error)
    assert payload["bot_id"] == "bot-prod"
    assert payload["metric_type"] == "account_snapshot"


def test_emit_metric_logs_debug_with_standard_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    error = ValueError("bad metric")
    store = FailingStore(error)
    logger = logging.getLogger("tests.telemetry")

    caplog.set_level(logging.DEBUG, logger.name)

    emit_metric(store, "bot-std", {"event_type": "system_check"}, logger=logger)

    assert any(
        "Failed to emit metric to event store: bad metric" in record.message
        for record in caplog.records
    )


def test_emit_metric_can_raise_when_requested() -> None:
    error = RuntimeError("fatal")
    store = FailingStore(error)

    with pytest.raises(RuntimeError, match="fatal"):
        emit_metric(store, "bot-raise", {"event_type": "panic"}, raise_on_error=True)
