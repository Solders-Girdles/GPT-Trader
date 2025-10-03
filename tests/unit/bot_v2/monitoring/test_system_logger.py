from __future__ import annotations

import json
from typing import Any

import pytest

from bot_v2.monitoring.system.logger import LogLevel, ProductionLogger


def _make_logger(
    monkeypatch: pytest.MonkeyPatch, *, enable_console: bool = False
) -> tuple[ProductionLogger, list[dict[str, Any]]]:
    logger = ProductionLogger(service_name="svc", enable_console=enable_console)
    emitted: list[dict[str, Any]] = []

    class DummyLogger:
        handlers = [object()]

        @staticmethod
        def log(level: int, message: str) -> None:  # noqa: ARG003
            emitted.append(json.loads(message))

    # Patch the emitter's logger (refactored structure)
    logger._emitter._py_logger = DummyLogger()  # type: ignore[attr-defined]
    return logger, emitted


def test_log_event_respects_min_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PERPS_MIN_LOG_LEVEL", "warning")
    monkeypatch.delenv("PERPS_DEBUG", raising=False)
    logger, emitted = _make_logger(monkeypatch)

    logger.log_event(LogLevel.INFO, "system_start", "Booting")
    assert emitted == []
    assert logger.get_recent_logs() == []

    logger.log_event(LogLevel.ERROR, "system_start", "Failure")
    assert emitted[0]["message"] == "Failure"


def test_log_error_adds_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PERPS_MIN_LOG_LEVEL", raising=False)
    logger, emitted = _make_logger(monkeypatch)

    err = ValueError("bad input")
    logger.log_error(err, context="trade_loop", order_id="123")

    record = emitted[-1]
    assert record["error_type"] == "ValueError"
    assert record["error_context"] == "trade_loop"
    assert record["order_id"] == "123"


def test_log_trade_and_correlation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    logger, emitted = _make_logger(monkeypatch)
    logger.set_correlation_id("abcd1234")
    logger.log_trade(
        "buy", "BTC-USD", quantity=1.0, price=50000.0, strategy="default", success=False
    )

    record = emitted[-1]
    assert record["correlation_id"] == "abcd1234"
    assert record["trade_action"] == "buy"
    assert record["success"] is False


def test_recent_logs_buffer_trims(monkeypatch: pytest.MonkeyPatch) -> None:
    logger, emitted = _make_logger(monkeypatch)
    logger._buffer._max_size = 3  # noqa: SLF001

    for idx in range(5):
        logger.log_event(LogLevel.INFO, "heartbeat", f"event-{idx}")

    recent = logger.get_recent_logs()
    assert len(recent) == 3
    assert recent[0]["message"] == "event-2"
    assert len(emitted) == 5


def test_log_performance_console_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PERPS_JSON_CONSOLE", "true")
    logger, emitted = _make_logger(monkeypatch, enable_console=True)

    logger.log_performance("load_config", duration_ms=12.5)

    stdout = capsys.readouterr().out.strip()
    assert stdout  # console output enabled
    assert emitted  # also emitted to structured logger


def test_log_ml_prediction_and_performance_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    logger, emitted = _make_logger(monkeypatch)
    logger.log_ml_prediction(
        "risk_model",
        prediction={"side": "long"},
        confidence=0.85,
        input_features={"vol": 0.1, "mean": 5, "skew": 0.2},
        inference_time_ms=4.2,
    )
    record = emitted[-1]
    assert record["event_type"] == "ml_prediction"
    assert record["feature_count"] == 3
    assert record["inference_time_ms"] == 4.2

    logger.log_performance("rebalance", duration_ms=55.0, success=False)
    record = emitted[-1]
    assert record["success"] is False
    assert "rebalance" in record["message"]


def test_auth_pnl_funding_and_market_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    logger, emitted = _make_logger(monkeypatch)

    logger.log_auth_event("jwt_refresh", provider="coinbase_cdp", success=False, error_code="401")
    logger.log_pnl(
        "BTC-USD", realized_pnl=10.5, unrealized_pnl=-2.0, fees=0.1, funding=0.2, position_size=0.3
    )
    logger.log_funding(
        "BTC-USD", funding_rate=0.0001, payment=-0.5, period_start="t0", period_end="t1"
    )
    logger.log_market_heartbeat(
        "coinbase",
        last_update_ts="2025-01-15T12:00:00Z",
        latency_ms=12.3,
        staleness_ms=5.6,
        threshold_ms=50,
    )

    events = [entry["event_type"] for entry in emitted[-4:]]
    assert events == ["auth_event", "pnl_update", "funding_applied", "market_heartbeat"]


def test_correlation_generation_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    logger, emitted = _make_logger(monkeypatch)
    generated_id = logger.get_correlation_id()
    assert generated_id

    logger.log_event(LogLevel.INFO, "status", "ready")
    assert emitted[-1]["correlation_id"] == generated_id

    # Check performance stats via public API
    stats = logger.get_performance_stats()
    assert stats["total_logs"] >= 1
    assert stats["total_log_time_ms"] > 0
