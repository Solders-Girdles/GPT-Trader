"""Tests for `TickerCache` in `coinbase/market_data_service.py`."""

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
from gpt_trader.utilities.datetime_helpers import utc_now


class TestTickerCache:
    """Tests for TickerCache class."""

    def test_cache_init_default_ttl(self) -> None:
        """Test default TTL is 5 seconds."""
        cache = TickerCache()
        assert cache.ttl == 5

    def test_cache_init_custom_ttl(self) -> None:
        """Test custom TTL can be set."""
        cache = TickerCache(ttl_seconds=30)
        assert cache.ttl == 30

    def test_cache_update_and_get(self) -> None:
        """Test updating and retrieving from cache."""
        cache = TickerCache()
        ticker = Ticker(
            symbol="BTC-USD",
            bid=49900.0,
            ask=50100.0,
            last=50000.0,
            ts=utc_now(),
        )

        cache.update(ticker)
        result = cache.get("BTC-USD")

        assert result is ticker

    def test_cache_get_nonexistent(self) -> None:
        """Test getting nonexistent symbol returns None."""
        cache = TickerCache()

        result = cache.get("NONEXISTENT")

        assert result is None

    def test_cache_update_overwrites(self) -> None:
        """Test that updating same symbol overwrites."""
        cache = TickerCache()
        ts1 = utc_now()
        ts2 = utc_now() + timedelta(seconds=1)

        ticker1 = Ticker(symbol="BTC-USD", bid=49000.0, ask=50000.0, last=49500.0, ts=ts1)
        ticker2 = Ticker(symbol="BTC-USD", bid=50000.0, ask=51000.0, last=50500.0, ts=ts2)

        cache.update(ticker1)
        cache.update(ticker2)

        result = cache.get("BTC-USD")
        assert result.last == 50500.0
        assert result.ts == ts2

    def test_cache_multiple_symbols(self) -> None:
        """Test caching multiple symbols."""
        cache = TickerCache()
        ts = utc_now()

        btc = Ticker(symbol="BTC-USD", bid=49900.0, ask=50100.0, last=50000.0, ts=ts)
        eth = Ticker(symbol="ETH-USD", bid=2990.0, ask=3010.0, last=3000.0, ts=ts)

        cache.update(btc)
        cache.update(eth)

        assert cache.get("BTC-USD") is btc
        assert cache.get("ETH-USD") is eth

    def test_is_stale_nonexistent_symbol(self) -> None:
        """Test that nonexistent symbol is considered stale."""
        cache = TickerCache()

        assert cache.is_stale("NONEXISTENT") is True

    def test_is_stale_fresh_ticker(self) -> None:
        """Test that recently updated ticker is not stale."""
        cache = TickerCache(ttl_seconds=10)
        ticker = Ticker(
            symbol="BTC-USD",
            bid=49900.0,
            ask=50100.0,
            last=50000.0,
            ts=utc_now(),
        )
        cache.update(ticker)

        assert cache.is_stale("BTC-USD") is False

    def test_is_stale_old_ticker(self) -> None:
        """Test that old ticker is considered stale."""
        cache = TickerCache(ttl_seconds=5)
        old_time = utc_now() - timedelta(seconds=10)
        ticker = Ticker(
            symbol="BTC-USD",
            bid=49900.0,
            ask=50100.0,
            last=50000.0,
            ts=old_time,
        )
        cache.update(ticker)

        assert cache.is_stale("BTC-USD") is True

    def test_cache_with_zero_ttl(self) -> None:
        """Test cache with zero TTL (always stale)."""
        cache = TickerCache(ttl_seconds=0)
        ticker = Ticker(
            symbol="BTC-USD",
            bid=50000.0,
            ask=50100.0,
            last=50050.0,
            ts=utc_now(),
        )
        cache.update(ticker)

        assert cache.is_stale("BTC-USD") is True

    def test_empty_symbol_string(self) -> None:
        """Test handling of empty symbol string."""
        cache = TickerCache()
        ticker = Ticker(symbol="", bid=100.0, ask=101.0, last=100.5, ts=utc_now())

        cache.update(ticker)
        result = cache.get("")

        assert result is ticker


class TestTickerServiceEdges:
    """Edge case tests for ticker service and cache."""

    def test_ticker_cache_stale_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = TickerCache(ttl_seconds=5)
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        cache.update(Ticker(symbol="BTC-USD", bid=1.0, ask=2.0, last=1.5, ts=base_time))

        monkeypatch.setattr(
            market_data_service,
            "utc_now",
            lambda: base_time + timedelta(seconds=5),
        )

        assert cache.is_stale("BTC-USD") is False

    def test_ticker_cache_stale_after_ttl(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = TickerCache(ttl_seconds=5)
        base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        cache.update(Ticker(symbol="BTC-USD", bid=1.0, ask=2.0, last=1.5, ts=base_time))

        monkeypatch.setattr(
            market_data_service,
            "utc_now",
            lambda: base_time + timedelta(seconds=6),
        )

        assert cache.is_stale("BTC-USD") is True

    def test_ticker_service_start_creates_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_ticker_service_stop_joins_thread(self) -> None:
        service = CoinbaseTickerService()
        service._running = True
        thread = Mock()
        service._thread = thread

        service.stop()

        assert service._running is False
        thread.join.assert_called_once_with(timeout=1.0)
