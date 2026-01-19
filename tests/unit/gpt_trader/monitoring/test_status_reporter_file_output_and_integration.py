"""Unit tests for StatusReporter file output."""

from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

import gpt_trader.monitoring.status_reporter as status_reporter
from gpt_trader.monitoring.status_reporter import StatusReporter


def _freeze_time(
    monkeypatch: pytest.MonkeyPatch,
    start: float = 1000.0,
) -> Callable[[float], None]:
    now = {"value": start}
    monkeypatch.setattr(status_reporter.time, "time", lambda: now["value"])

    def advance(seconds: float) -> None:
        now["value"] += seconds

    return advance


@pytest.mark.asyncio
async def test_writes_valid_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    advance = _freeze_time(monkeypatch, start=1000.0)
    status_file = tmp_path / "status.json"
    reporter = StatusReporter(
        status_file=str(status_file),
        bot_id="test-bot",
        observer_interval=0.1,
        file_write_interval=1.0,
    )
    reporter._running = True
    reporter._start_time = 990.0

    reporter.update_price("BTC-USD", Decimal("50000.123"))
    reporter.update_positions({"BTC-PERP": {"quantity": Decimal("1"), "unrealized_pnl": "5"}})
    reporter.record_cycle()
    advance(1.0)

    await reporter._write_status()

    data = json.loads(status_file.read_text())
    assert data["bot_id"] == "test-bot"
    assert data["engine"]["running"] is True
    assert data["engine"]["cycle_count"] == 1
    assert data["market"]["last_prices"]["BTC-USD"] == "50000.123"
    assert data["positions"]["count"] == 1
    assert data["positions"]["positions"]["BTC-PERP"]["unrealized_pnl"] == "5"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_handles_decimal_serialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _freeze_time(monkeypatch, start=2000.0)
    status_file = tmp_path / "status.json"
    reporter = StatusReporter(status_file=str(status_file))
    reporter._running = True

    reporter.update_price("BTC-USD", Decimal("50000.12345678"))
    reporter.update_positions({"BTC-PERP": {"unrealized_pnl": Decimal("123.456")}})

    await reporter._write_status()

    data = json.loads(status_file.read_text())
    assert data["market"]["last_prices"]["BTC-USD"] == "50000.12345678"
    assert data["positions"]["positions"]["BTC-PERP"]["unrealized_pnl"] == "123.456"


@pytest.mark.asyncio
async def test_atomic_write_overwrites_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    advance = _freeze_time(monkeypatch, start=3000.0)
    status_file = tmp_path / "status.json"
    reporter = StatusReporter(status_file=str(status_file))
    reporter._running = True
    reporter._start_time = 2990.0

    for _ in range(3):
        reporter.record_cycle()
        await reporter._write_status()
        data = json.loads(status_file.read_text())
        assert data["engine"]["cycle_count"] >= 1
        assert not list(tmp_path.glob(".status_*.tmp"))
        advance(1.0)
