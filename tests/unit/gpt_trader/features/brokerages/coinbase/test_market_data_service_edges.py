from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

import gpt_trader.features.brokerages.coinbase.market_data_service as market_data_service
from gpt_trader.features.brokerages.coinbase.market_data_service import (
    CoinbaseTickerService,
    Ticker,
    TickerCache,
)


def test_ticker_cache_stale_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TickerCache(ttl_seconds=5)
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    cache.update(Ticker(symbol="BTC-USD", bid=1.0, ask=2.0, last=1.5, ts=base_time))

    monkeypatch.setattr(
        market_data_service,
        "utc_now",
        lambda: base_time + timedelta(seconds=5),
    )

    assert cache.is_stale("BTC-USD") is False


def test_ticker_cache_stale_after_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = TickerCache(ttl_seconds=5)
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    cache.update(Ticker(symbol="BTC-USD", bid=1.0, ask=2.0, last=1.5, ts=base_time))

    monkeypatch.setattr(
        market_data_service,
        "utc_now",
        lambda: base_time + timedelta(seconds=6),
    )

    assert cache.is_stale("BTC-USD") is True


def test_ticker_service_start_creates_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    service = CoinbaseTickerService()
    thread = Mock()
    thread.start = Mock()

    def fake_thread(*_args, **kwargs):
        assert kwargs["target"] == service._run
        return thread

    monkeypatch.setattr(market_data_service.threading, "Thread", fake_thread)

    service.start()

    assert service._running is True
    assert service._thread is thread
    thread.start.assert_called_once_with()


def test_ticker_service_stop_joins_thread() -> None:
    service = CoinbaseTickerService()
    service._running = True
    thread = Mock()
    service._thread = thread

    service.stop()

    assert service._running is False
    thread.join.assert_called_once_with(timeout=1.0)
