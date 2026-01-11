"""Edge-case unit tests for tracing utilities."""

from __future__ import annotations

from typing import Any

import gpt_trader.observability.tracing as tracing


class _DummySpanContext:
    def __init__(self, name: str, attributes: dict[str, Any], record: dict[str, Any]) -> None:
        self._name = name
        self._attributes = attributes
        self._record = record

    def __enter__(self) -> object:
        self._record["name"] = self._name
        self._record["attributes"] = self._attributes
        return object()

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _DummyTracer:
    def __init__(self, record: dict[str, Any]) -> None:
        self._record = record

    def start_as_current_span(self, name: str, attributes: dict[str, Any] | None = None):
        return _DummySpanContext(name, attributes or {}, self._record)


def test_trace_span_stringifies_context_and_merges_attributes(monkeypatch) -> None:
    record: dict[str, Any] = {}

    monkeypatch.setattr(tracing, "_tracing_enabled", True)
    monkeypatch.setattr(tracing, "_tracer", _DummyTracer(record))
    monkeypatch.setattr(
        tracing,
        "get_log_context",
        lambda: {"cycle": 1, "meta": {"a": 1}, "flags": [1, 2], "skip": None},
    )

    with tracing.trace_span("cycle", {"cycle": 2, "tags": ["alpha"]}) as span:
        assert span is not None

    attrs = record["attributes"]
    assert attrs["cycle"] == 2
    assert attrs["meta"] == "{'a': 1}"
    assert attrs["flags"] == "[1, 2]"
    assert attrs["tags"] == "['alpha']"
    assert "skip" not in attrs


def test_init_tracing_disabled_clears_tracer(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_tracer", object())
    monkeypatch.setattr(tracing, "_tracing_enabled", True)

    result = tracing.init_tracing(enabled=False)

    assert result is False
    assert tracing._tracer is None
    assert tracing._tracing_enabled is False


def test_failed_init_when_otel_missing_disables_tracing(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    monkeypatch.setattr(tracing, "_tracer", object())
    monkeypatch.setattr(tracing, "_tracing_enabled", True)

    result = tracing.init_tracing(enabled=True)

    assert result is False
    assert tracing.get_tracer() is None
    assert tracing.is_tracing_enabled() is False


def test_trace_span_yields_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(tracing, "_tracing_enabled", False)
    monkeypatch.setattr(tracing, "_tracer", _DummyTracer({}))

    with tracing.trace_span("noop", {"attr": "value"}) as span:
        assert span is None
