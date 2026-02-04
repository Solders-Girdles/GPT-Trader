"""Unit tests for StatusReporter file output."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import pytest

import gpt_trader.monitoring.status_reporter as status_reporter
from gpt_trader.monitoring.status_reporter import (
    MAX_TICKER_FRESHNESS_SYMBOLS,
    DecimalEncoder,
    StatusReporter,
)


class _TestClock:
    def __init__(self, start: float) -> None:
        self._now = start

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _freeze_time(
    monkeypatch: pytest.MonkeyPatch,
    start: float = 1000.0,
) -> Callable[[float], None]:
    clock = _TestClock(start)
    monkeypatch.setattr(status_reporter, "get_clock", lambda: clock)
    return clock.advance


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


@pytest.mark.asyncio
async def test_write_status_creates_directories_and_writes_json(tmp_path: Path) -> None:
    status_file = tmp_path / "missing" / "dir" / "status.json"
    reporter = StatusReporter(status_file=str(status_file))

    await reporter._write_status_to_file()

    assert status_file.exists()
    data = json.loads(status_file.read_text())
    assert "engine" in data
    assert "market" in data


def test_decimal_encoder_handles_nested_decimals() -> None:
    reporter = StatusReporter()
    reporter.update_positions(
        {
            "BTC-PERP": {
                "quantity": Decimal("1.5"),
                "unrealized_pnl": Decimal("0.1"),
                "meta": {"nested": Decimal("2.5")},
            }
        }
    )
    reporter._update_status()
    payload = reporter._serialize_status()

    encoded = json.dumps(payload, cls=DecimalEncoder)
    decoded = json.loads(encoded)
    assert decoded["positions"]["positions"]["BTC-PERP"]["meta"]["nested"] == "2.5"


@pytest.mark.asyncio
async def test_write_status_handles_write_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    status_file = tmp_path / "status.json"
    reporter = StatusReporter(status_file=str(status_file))

    monkeypatch.setattr(
        json, "dump", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom"))
    )

    await reporter._write_status_to_file()

    assert not status_file.exists()


def test_serialize_status_includes_expected_keys() -> None:
    reporter = StatusReporter()
    payload = reporter._serialize_status()

    for key in (
        "bot_id",
        "timestamp",
        "engine",
        "market",
        "positions",
        "orders",
        "trades",
        "account",
        "strategy",
        "risk",
        "system",
        "heartbeat",
        "websocket",
        "ticker_freshness",
    ):
        assert key in payload


class _FakeTickerCache:
    def __init__(self, stale_symbols: set[str]) -> None:
        self._stale_symbols = stale_symbols

    def is_stale(self, symbol: str) -> bool:
        return symbol in self._stale_symbols


class _FakeMarketDataService:
    def __init__(self, symbols: list[str], stale_symbols: set[str]) -> None:
        self.symbols = symbols
        self.ticker_cache = _FakeTickerCache(stale_symbols)


def test_ticker_freshness_summary_includes_counts_and_reason() -> None:
    reporter = StatusReporter()
    service = _FakeMarketDataService(
        symbols=["BTC-USD", "ETH-USD", "SOL-USD"],
        stale_symbols={"BTC-USD", "SOL-USD"},
    )
    reporter.set_market_data_service(service)

    reporter._update_status()
    payload = reporter._serialize_status()
    summary = payload["ticker_freshness"]

    assert summary["symbol_count"] == 3
    assert summary["stale_count"] == 2
    assert summary["severity"] == "warning"
    assert summary["reason"] == "stale_symbols_detected"
    assert summary["status"] == "fail"
    assert summary["stale_symbols"] == ["BTC-USD", "SOL-USD"]


def test_ticker_freshness_summary_caps_symbols() -> None:
    reporter = StatusReporter()
    symbols = [f"SYM-{i}" for i in range(MAX_TICKER_FRESHNESS_SYMBOLS + 2)]
    service = _FakeMarketDataService(
        symbols=symbols,
        stale_symbols=set(symbols),
    )
    reporter.set_market_data_service(service)

    reporter._update_status()
    payload = reporter._serialize_status()
    summary = payload["ticker_freshness"]

    assert summary["stale_count"] == len(symbols)
    assert summary["stale_symbols_capped"] is True
    assert len(summary["stale_symbols"]) == MAX_TICKER_FRESHNESS_SYMBOLS


class TestStatusReporterStart:
    """Tests for StatusReporter start method."""

    @pytest.mark.asyncio
    async def test_start_when_disabled(self) -> None:
        reporter = StatusReporter(enabled=False)
        task = await reporter.start()
        assert task is None
        assert reporter._running is False

    @pytest.mark.asyncio
    async def test_start_creates_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                file_write_interval=1,
            )
            task = await reporter.start()

            try:
                assert task is not None
                assert reporter._running is True
                assert reporter._start_time > 0
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_start_writes_initial_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                bot_id="test-bot",
            )
            await reporter.start()

            try:
                assert status_file.exists()
                data = json.loads(status_file.read_text())
                assert data["bot_id"] == "test-bot"
                assert "timestamp" in data
            finally:
                await reporter.stop()

    @pytest.mark.asyncio
    async def test_start_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "subdir" / "status.json"
            reporter = StatusReporter(status_file=str(status_file))

            await reporter.start()

            try:
                assert status_file.parent.exists()
                assert status_file.exists()
            finally:
                await reporter.stop()


class TestStatusReporterStop:
    """Tests for StatusReporter stop method."""

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            status_file = Path(tmpdir) / "status.json"
            reporter = StatusReporter(
                status_file=str(status_file),
                file_write_interval=10,
            )
            await reporter.start()
            await reporter.stop()

            assert reporter._running is False
            assert reporter._task is None
