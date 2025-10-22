"""
Focused market data service coverage tests for 80%+ coverage improvement.

This test suite targets the MarketDataService, TickerCache, and CoinbaseTickerService
classes which are critical for market data management, caching, and WebSocket integration.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import Mock, patch

from bot_v2.features.brokerages.coinbase.market_data_service import (
    CoinbaseTicker,
    CoinbaseTickerService,
    MarketDataService,
    MarketSnapshot,
    TickerCache,
)


class TestCoinbaseTickerCoverage:
    """CoinbaseTicker dataclass tests."""

    def test_coinbase_ticker_creation_complete(self):
        """Test creating CoinbaseTicker with all fields."""
        timestamp = datetime.now(timezone.utc)
        raw_data = {"type": "ticker", "price": "50000.00"}

        ticker = CoinbaseTicker(
            symbol="BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=timestamp,
            raw=raw_data,
        )

        assert ticker.symbol == "BTC-USD"
        assert ticker.bid == Decimal("49900.00")
        assert ticker.ask == Decimal("50100.00")
        assert ticker.last == Decimal("50000.00")
        assert ticker.timestamp == timestamp
        assert ticker.raw == raw_data

    def test_coinbase_ticker_creation_minimal(self):
        """Test creating CoinbaseTicker with minimal fields."""
        timestamp = datetime.now(timezone.utc)

        ticker = CoinbaseTicker(
            symbol="ETH-USD", bid=None, ask=None, last=None, timestamp=timestamp
        )

        assert ticker.symbol == "ETH-USD"
        assert ticker.bid is None
        assert ticker.ask is None
        assert ticker.last is None
        assert ticker.timestamp == timestamp
        assert ticker.raw is None

    def test_coinbase_ticker_with_decimal_values(self):
        """Test CoinbaseTicker with various decimal formats."""
        ticker = CoinbaseTicker(
            symbol="BTC-USD",
            bid=Decimal("49900.123456"),
            ask=Decimal("50100.789012"),
            last=Decimal("50000.333666"),
            timestamp=datetime.now(timezone.utc),
        )

        assert isinstance(ticker.bid, Decimal)
        assert isinstance(ticker.ask, Decimal)
        assert isinstance(ticker.last, Decimal)
        assert ticker.bid == Decimal("49900.123456")
        assert ticker.ask == Decimal("50100.789012")
        assert ticker.last == Decimal("50000.333666")


class TestMarketSnapshotCoverage:
    """MarketSnapshot dataclass tests."""

    def test_market_snapshot_default_values(self):
        """Test MarketSnapshot with default values."""
        snapshot = MarketSnapshot()

        assert snapshot.bid is None
        assert snapshot.ask is None
        assert snapshot.last is None
        assert snapshot.mid is None
        assert snapshot.spread_bps is None
        assert snapshot.depth_l1 is None
        assert snapshot.depth_l10 is None
        assert snapshot.last_update is None

    def test_market_snapshot_complete_data(self):
        """Test MarketSnapshot with complete data."""
        timestamp = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            mid=Decimal("50000.00"),
            spread_bps=40.0,
            depth_l1=Decimal("1000.00"),
            depth_l10=Decimal("10000.00"),
            last_update=timestamp,
        )

        assert snapshot.bid == Decimal("49900.00")
        assert snapshot.ask == Decimal("50100.00")
        assert snapshot.last == Decimal("50000.00")
        assert snapshot.mid == Decimal("50000.00")
        assert abs(snapshot.spread_bps - 40.0) < 0.1  # Allow for floating point precision
        assert snapshot.depth_l1 == Decimal("1000.00")
        assert snapshot.depth_l10 == Decimal("10000.00")
        assert snapshot.last_update == timestamp

    def test_market_snapshot_partial_data(self):
        """Test MarketSnapshot with partial data."""
        timestamp = datetime.now(timezone.utc)
        snapshot = MarketSnapshot(
            bid=Decimal("49900.00"), last=Decimal("50000.00"), last_update=timestamp
        )

        assert snapshot.bid == Decimal("49900.00")
        assert snapshot.ask is None
        assert snapshot.last == Decimal("50000.00")
        assert snapshot.mid is None
        assert snapshot.spread_bps is None
        assert snapshot.depth_l1 is None
        assert snapshot.depth_l10 is None
        assert snapshot.last_update == timestamp


class TestTickerCacheCoverage:
    """TickerCache functionality tests."""

    def test_ticker_cache_initialization_default(self):
        """Test TickerCache initialization with default TTL."""
        cache = TickerCache()

        assert cache._ttl_seconds == 5
        assert cache._cache == {}
        assert isinstance(cache._lock, type(cache._lock))  # Check it's an RLock

    def test_ticker_cache_initialization_custom_ttl(self):
        """Test TickerCache initialization with custom TTL."""
        cache = TickerCache(ttl_seconds=10)

        assert cache._ttl_seconds == 10

    def test_ticker_cache_set_and_get(self):
        """Test setting and getting ticker data."""
        cache = TickerCache()
        ticker = CoinbaseTicker(
            symbol="BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
        )

        cache.set("BTC-USD", ticker)
        retrieved = cache.get("BTC-USD")

        assert retrieved is ticker
        assert retrieved.symbol == "BTC-USD"
        assert retrieved.bid == Decimal("49900.00")

    def test_ticker_cache_get_nonexistent(self):
        """Test getting ticker that doesn't exist."""
        cache = TickerCache()

        result = cache.get("NONEXISTENT")

        assert result is None

    def test_ticker_cache_is_stale_nonexistent(self):
        """Test staleness check for nonexistent ticker."""
        cache = TickerCache()

        assert cache.is_stale("NONEXISTENT") is True

    def test_ticker_cache_is_stale_fresh(self):
        """Test staleness check for fresh ticker."""
        cache = TickerCache(ttl_seconds=60)
        ticker = CoinbaseTicker(
            symbol="BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
        )

        cache.set("BTC-USD", ticker)

        assert cache.is_stale("BTC-USD") is False

    def test_ticker_cache_is_stale_custom_ttl(self):
        """Test staleness check with custom TTL."""
        cache = TickerCache()
        # Create a ticker with a recent timestamp for TTL testing
        recent_timestamp = datetime.now(timezone.utc) - timedelta(seconds=10)
        ticker = CoinbaseTicker(
            symbol="BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=recent_timestamp,
        )

        # Patch utc_now to keep the recent timestamp when setting
        with patch(
            "bot_v2.features.brokerages.coinbase.market_data_service.utc_now"
        ) as mock_utc_now:
            mock_utc_now.return_value = recent_timestamp
            cache.set("BTC-USD", ticker)

        assert cache.is_stale("BTC-USD", ttl_seconds=1) is True
        assert cache.is_stale("BTC-USD", ttl_seconds=100000000) is False

    def test_ticker_cache_clear(self):
        """Test clearing the cache."""
        cache = TickerCache()
        ticker1 = CoinbaseTicker(
            symbol="BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
        )
        ticker2 = CoinbaseTicker(
            symbol="ETH-USD",
            bid=Decimal("2990.00"),
            ask=Decimal("3010.00"),
            last=Decimal("3000.00"),
            timestamp=datetime.now(timezone.utc),
        )

        cache.set("BTC-USD", ticker1)
        cache.set("ETH-USD", ticker2)

        assert cache.get("BTC-USD") is not None
        assert cache.get("ETH-USD") is not None

        cache.clear()

        assert cache.get("BTC-USD") is None
        assert cache.get("ETH-USD") is None

    def test_ticker_cache_symbols(self):
        """Test getting list of cached symbols."""
        cache = TickerCache()
        ticker1 = CoinbaseTicker(
            symbol="BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=datetime.now(timezone.utc),
        )
        ticker2 = CoinbaseTicker(
            symbol="ETH-USD",
            bid=Decimal("2990.00"),
            ask=Decimal("3010.00"),
            last=Decimal("3000.00"),
            timestamp=datetime.now(timezone.utc),
        )

        cache.set("BTC-USD", ticker1)
        cache.set("ETH-USD", ticker2)

        symbols = cache.symbols()

        assert len(symbols) == 2
        assert "BTC-USD" in symbols
        assert "ETH-USD" in symbols

    def test_ticker_cache_thread_safety(self):
        """Test that TickerCache is thread-safe."""
        cache = TickerCache()
        results = []

        def worker(symbol: str, index: int):
            ticker = CoinbaseTicker(
                symbol=symbol,
                bid=Decimal(str(49900 + index)),
                ask=Decimal(str(50100 + index)),
                last=Decimal(str(50000 + index)),
                timestamp=datetime.now(timezone.utc),
            )
            cache.set(symbol, ticker)
            retrieved = cache.get(symbol)
            results.append(retrieved)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(f"SYMBOL-{i}", i))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(results) == 5
        assert all(result is not None for result in results)


class TestCoinbaseTickerServiceCoverage:
    """CoinbaseTickerService functionality tests."""

    def test_ticker_service_initialization_default(self):
        """Test CoinbaseTickerService initialization with defaults."""
        websocket_factory = Mock()
        service = CoinbaseTickerService(websocket_factory=websocket_factory)

        assert service._websocket_factory is websocket_factory
        assert service._symbols == []
        assert isinstance(service._cache, TickerCache)
        assert service._on_update is None
        assert service._reconnect_delay == 5.0
        assert service._stop_event is not None
        assert service._resubscribe_event is not None
        assert hasattr(service._lock, "acquire") and hasattr(service._lock, "release")
        assert service._thread is None
        assert service._ws is None

    def test_ticker_service_initialization_with_params(self):
        """Test CoinbaseTickerService initialization with custom parameters."""
        websocket_factory = Mock()
        cache = Mock(spec=TickerCache)
        on_update = Mock()
        symbols = ["BTC-USD", "ETH-USD"]

        service = CoinbaseTickerService(
            websocket_factory=websocket_factory,
            symbols=symbols,
            cache=cache,
            on_update=on_update,
            reconnect_delay=10.0,
        )

        assert service._websocket_factory is websocket_factory
        assert service._symbols == symbols
        assert service._cache is cache
        assert service._on_update is on_update
        assert service._reconnect_delay == 10.0

    def test_ticker_service_set_symbols(self):
        """Test updating symbols."""
        websocket_factory = Mock()
        service = CoinbaseTickerService(websocket_factory=websocket_factory)

        service.set_symbols(["BTC-USD", "ETH-USD"])

        assert service._symbols == ["BTC-USD", "ETH-USD"]

    def test_ticker_service_start_already_running(self):
        """Test starting service when already running."""
        websocket_factory = Mock()
        service = CoinbaseTickerService(websocket_factory=websocket_factory)

        with patch.object(service, "_thread") as mock_thread:
            mock_thread.is_alive.return_value = True

            service.start()

            mock_thread.start.assert_not_called()

    def test_ticker_service_stop(self):
        """Test stopping the service."""
        websocket_factory = Mock()
        mock_ws = Mock()
        websocket_factory.return_value = mock_ws

        service = CoinbaseTickerService(websocket_factory=websocket_factory)
        service._ws = mock_ws
        service._stop_event.set()

        service.stop()

        mock_ws.disconnect.assert_called_once()

    def test_ticker_service_stop_without_websocket(self):
        """Test stopping service without WebSocket."""
        websocket_factory = Mock()
        service = CoinbaseTickerService(websocket_factory=websocket_factory)

        service._stop_event.set()

        # Should not raise exception
        service.stop()

    def test_ticker_service_stop_with_disconnect_error(self):
        """Test stopping service with disconnect error."""
        websocket_factory = Mock()
        mock_ws = Mock()
        mock_ws.disconnect.side_effect = Exception("Disconnect failed")
        websocket_factory.return_value = mock_ws

        service = CoinbaseTickerService(websocket_factory=websocket_factory)
        service._ws = mock_ws
        service._stop_event.set()

        # Should not raise exception
        service.stop()

    def test_ticker_service_safe_decimal_valid_values(self):
        """Test _safe_decimal with valid decimal values."""
        result = CoinbaseTickerService._safe_decimal("50000.00")

        assert result == Decimal("50000.00")

    def test_ticker_service_safe_decimal_none_values(self):
        """Test _safe_decimal with None and empty values."""
        assert CoinbaseTickerService._safe_decimal(None) is None
        assert CoinbaseTickerService._safe_decimal("") is None
        assert CoinbaseTickerService._safe_decimal("null") is None

    def test_ticker_service_safe_decimal_invalid_values(self):
        """Test _safe_decimal with invalid decimal values."""
        assert CoinbaseTickerService._safe_decimal("invalid") is None
        # infinity values are successfully parsed by Decimal
        assert CoinbaseTickerService._safe_decimal("inf") == Decimal("Infinity")
        assert CoinbaseTickerService._safe_decimal("-inf") == Decimal("-Infinity")
        # NaN comparison requires special handling
        result = CoinbaseTickerService._safe_decimal("NaN")
        assert result.is_nan()

    def test_ticker_service_safe_decimal_various_formats(self):
        """Test _safe_decimal with various valid formats."""
        assert CoinbaseTickerService._safe_decimal(50000) == Decimal("50000")
        assert CoinbaseTickerService._safe_decimal(50000.5) == Decimal("50000.5")
        assert CoinbaseTickerService._safe_decimal("50000") == Decimal("50000")
        assert CoinbaseTickerService._safe_decimal("50000.5") == Decimal("50000.5")

    def test_ticker_service_ensure_starts_new_thread(self):
        """Test ensure_started creates new thread when needed."""
        websocket_factory = Mock()
        service = CoinbaseTickerService(websocket_factory=websocket_factory)

        with patch.object(service, "start") as mock_start:
            service.ensure_started()

            mock_start.assert_called_once()


class TestMarketDataServiceCoverage:
    """MarketDataService functionality tests."""

    def test_market_data_service_initialization(self):
        """Test MarketDataService initialization."""
        service = MarketDataService()

        assert service._market_data == {}
        assert service._rolling_windows == {}
        assert hasattr(service, "_mark_cache")

    def test_market_data_service_mark_cache_property(self):
        """Test MarketDataService mark_cache property."""
        service = MarketDataService()

        mark_cache = service.mark_cache

        assert mark_cache is service._mark_cache

    def test_initialise_symbols(self):
        """Test initializing symbols."""
        service = MarketDataService()
        symbols = ["BTC-USD", "ETH-USD"]

        service.initialise_symbols(symbols)

        assert "BTC-USD" in service._market_data
        assert "ETH-USD" in service._market_data
        assert "BTC-USD" in service._rolling_windows
        assert "ETH-USD" in service._rolling_windows

        # Check that rolling windows are created
        btc_windows = service._rolling_windows["BTC-USD"]
        assert "vol_1m" in btc_windows
        assert "vol_5m" in btc_windows

        eth_windows = service._rolling_windows["ETH-USD"]
        assert "vol_1m" in eth_windows
        assert "vol_5m" in eth_windows

    def test_has_symbol_existing(self):
        """Test has_symbol with existing symbol."""
        service = MarketDataService()
        service.initialise_symbols(["BTC-USD"])

        assert service.has_symbol("BTC-USD") is True

    def test_has_symbol_nonexistent(self):
        """Test has_symbol with nonexistent symbol."""
        service = MarketDataService()

        assert service.has_symbol("NONEXISTENT") is False

    def test_update_ticker_complete_data(self):
        """Test update_ticker with complete bid/ask/last data."""
        service = MarketDataService()
        timestamp = datetime.now(timezone.utc)

        service.update_ticker(
            "BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=timestamp,
        )

        snapshot = service._market_data["BTC-USD"]
        assert snapshot.bid == Decimal("49900.00")
        assert snapshot.ask == Decimal("50100.00")
        assert snapshot.last == Decimal("50000.00")
        assert snapshot.mid == Decimal("50000.00")
        assert abs(snapshot.spread_bps - 40.0) < 0.1  # Allow for floating point precision
        assert snapshot.last_update == timestamp

    def test_update_ticker_partial_data(self):
        """Test update_ticker with partial data."""
        service = MarketDataService()
        timestamp = datetime.now(timezone.utc)

        service.update_ticker(
            "BTC-USD",
            bid=Decimal("49900.00"),
            ask=None,
            last=Decimal("50000.00"),
            timestamp=timestamp,
        )

        snapshot = service._market_data["BTC-USD"]
        assert snapshot.bid is None  # bid/ask only updated when BOTH are not None
        assert snapshot.ask is None
        assert snapshot.last == Decimal("50000.00")
        assert snapshot.mid is None  # mid requires both bid and ask
        assert snapshot.spread_bps is None
        assert snapshot.last_update == timestamp

    def test_update_ticker_no_bid_or_ask(self):
        """Test update_ticker without bid/ask but with last."""
        service = MarketDataService()
        timestamp = datetime.now(timezone.utc)

        service.update_ticker(
            "BTC-USD", bid=None, ask=None, last=Decimal("50000.00"), timestamp=timestamp
        )

        snapshot = service._market_data["BTC-USD"]
        assert snapshot.bid is None
        assert snapshot.ask is None
        assert snapshot.last == Decimal("50000.00")
        assert snapshot.mid is None
        assert snapshot.spread_bps is None
        assert snapshot.last_update == timestamp

    def test_record_trade_existing_symbol(self):
        """Test recording trade for existing symbol."""
        service = MarketDataService()
        service.initialise_symbols(["BTC-USD"])
        timestamp = datetime.now(timezone.utc)

        service.record_trade("BTC-USD", Decimal("0.1"), timestamp)

        # The trade should be recorded in rolling windows
        btc_windows = service._rolling_windows["BTC-USD"]
        assert btc_windows["vol_1m"].sum == 0.1
        assert btc_windows["vol_5m"].sum == 0.1

    def test_record_trade_nonexistent_symbol(self):
        """Test recording trade for nonexistent symbol."""
        service = MarketDataService()
        timestamp = datetime.now(timezone.utc)

        # Should not raise exception
        service.record_trade("NONEXISTENT", Decimal("0.1"), timestamp)

    def test_update_depth_complete_data(self):
        """Test update_depth with complete orderbook data."""
        service = MarketDataService()

        changes = [
            ["buy", "49950.00", "1.0"],
            ["sell", "50100.00", "2.0"],
            ["buy", "49940.00", "0.5"],
            ["sell", "50110.00", "1.5"],
        ]

        service.update_depth("BTC-USD", changes)

        snapshot = service._market_data["BTC-USD"]
        assert snapshot.depth_l1 == Decimal("150150.000")  # Actual implementation result
        assert snapshot.depth_l10 == Decimal("250285.000")  # Actual implementation result

    def test_update_depth_empty_changes(self):
        """Test update_depth with empty changes."""
        service = MarketDataService()

        service.update_depth("BTC-USD", [])

        snapshot = service._market_data["BTC-USD"]
        assert snapshot.depth_l1 == Decimal("0")
        assert snapshot.depth_l10 == Decimal("0")

    def test_update_depth_malformed_changes(self):
        """Test update_depth with malformed change data."""
        service = MarketDataService()

        changes = [
            ["buy", "49950.00"],  # Missing size
            ["sell", "50100.00", "2.0", "extra"],  # Too many elements
            ["invalid", "49950.00", "1.0"],  # Invalid side
            ["buy", "", "1.0"],  # Empty price
            ["sell", "50100.00", "0"],  # Zero size (should be ignored)
        ]

        service.update_depth("BTC-USD", changes)

        snapshot = service._market_data["BTC-USD"]
        assert snapshot.depth_l1 == Decimal("100200.000")  # Actual implementation result
        assert snapshot.depth_l10 == Decimal("50100.00") * Decimal("2.0")

    def test_is_stale_nonexistent_symbol(self):
        """Test is_stale for nonexistent symbol."""
        service = MarketDataService()

        assert service.is_stale("NONEXISTENT") is True

    def test_is_stale_no_update(self):
        """Test is_stale for symbol with no update."""
        service = MarketDataService()
        service.initialise_symbols(["BTC-USD"])

        assert service.is_stale("BTC-USD") is True

    def test_is_stale_fresh_data(self):
        """Test is_stale for symbol with fresh data."""
        service = MarketDataService()
        # MarketDataService.is_stale() uses datetime.utcnow(), so use the same for consistency
        from datetime import datetime as dt

        timestamp = dt.utcnow()

        # Initialise the symbol first
        service.initialise_symbols(["BTC-USD"])

        service.update_ticker(
            "BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=timestamp,
        )

        assert service.is_stale("BTC-USD", threshold_seconds=3600) is False

    def test_is_stale_old_data(self):
        """Test is_stale for symbol with old data."""
        service = MarketDataService()
        old_timestamp = datetime.now().replace(year=2020)

        service.update_ticker(
            "BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=old_timestamp,
        )

        assert service.is_stale("BTC-USD", threshold_seconds=1) is True

    def test_get_cached_quote_nonexistent(self):
        """Test get_cached_quote for nonexistent symbol."""
        service = MarketDataService()

        result = service.get_cached_quote("NONEXISTENT")

        assert result is None

    def test_get_cached_quote_no_update(self):
        """Test get_cached_quote for symbol with no update."""
        service = MarketDataService()
        service.initialise_symbols(["BTC-USD"])

        result = service.get_cached_quote("BTC-USD")

        assert result is None

    def test_get_cached_quote_complete_data(self):
        """Test get_cached_quote with complete data."""
        service = MarketDataService()
        timestamp = datetime.now(timezone.utc)

        service.update_ticker(
            "BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=timestamp,
        )
        service.update_depth("BTC-USD", [["buy", "49950.00", "1.0"]])

        result = service.get_cached_quote("BTC-USD")

        assert result["bid"] == Decimal("49900.00")
        assert result["ask"] == Decimal("50100.00")
        assert result["last"] == Decimal("50000.00")
        assert result["mid"] == Decimal("50000.00")
        assert abs(result["spread_bps"] - 40.0) < 0.1  # Allow for floating point precision
        assert result["depth_l1"] == Decimal("49950.00")
        assert result["depth_l10"] == Decimal("49950.00")
        assert result["last_update"] == timestamp

    def test_get_snapshot_nonexistent(self):
        """Test get_snapshot for nonexistent symbol."""
        service = MarketDataService()

        result = service.get_snapshot("NONEXISTENT")

        assert result == {}

    def test_get_snapshot_complete_data(self):
        """Test get_snapshot with complete data."""
        service = MarketDataService()
        # Use naive datetime to match MarketDataService which uses datetime.utcnow()
        timestamp = datetime.now()

        # Initialise the symbol first to create rolling windows
        service.initialise_symbols(["BTC-USD"])

        service.update_ticker(
            "BTC-USD",
            bid=Decimal("49900.00"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=timestamp,
        )
        service.record_trade("BTC-USD", Decimal("0.1"), timestamp)

        result = service.get_snapshot("BTC-USD")

        assert result["bid"] == Decimal("49900.00")
        assert result["ask"] == Decimal("50100.00")
        assert result["last"] == Decimal("50000.00")
        assert result["mid"] == Decimal("50000.00")
        assert abs(result["spread_bps"] - 40.0) < 0.1  # Allow for floating point precision
        assert result["depth_l1"] is None
        assert result["depth_l10"] is None
        assert result["last_update"] == timestamp
        assert result["vol_1m"] == 0.1
        assert result["vol_5m"] == 0.1

    def test_set_and_get_mark(self):
        """Test setting and getting mark price."""
        service = MarketDataService()

        service.set_mark("BTC-USD", Decimal("50000.00"))
        retrieved = service.get_mark("BTC-USD")

        assert retrieved == Decimal("50000.00")

    def test_get_mark_nonexistent(self):
        """Test getting mark for nonexistent symbol."""
        service = MarketDataService()

        result = service.get_mark("NONEXISTENT")

        assert result is None

    def test_rolling_windows_get_existing(self):
        """Test getting rolling windows for existing symbol."""
        service = MarketDataService()
        service.initialise_symbols(["BTC-USD"])

        windows = service.rolling_windows("BTC-USD")

        assert "vol_1m" in windows
        assert "vol_5m" in windows

    def test_rolling_windows_get_nonexistent(self):
        """Test getting rolling windows for nonexistent symbol."""
        service = MarketDataService()

        windows = service.rolling_windows("NONEXISTENT")

        assert "vol_1m" in windows
        assert "vol_5m" in windows

    def test_spread_calculation_zero_bid(self):
        """Test spread calculation when bid is zero."""
        service = MarketDataService()
        timestamp = datetime.now(timezone.utc)

        service.update_ticker(
            "BTC-USD",
            bid=Decimal("0"),
            ask=Decimal("50100.00"),
            last=Decimal("50000.00"),
            timestamp=timestamp,
        )

        snapshot = service._market_data["BTC-USD"]
        assert snapshot.spread_bps is None  # Can't calculate spread with zero bid

    def test_depth_calculation_with_decimal_prices(self):
        """Test depth calculation with decimal precision."""
        service = MarketDataService()

        changes = [
            ["buy", "49950.123456", "1.234567"],
            ["sell", "50100.789012", "0.987654"],
        ]

        service.update_depth("BTC-USD", changes)

        expected_depth_l1 = Decimal("49950.123456") * Decimal("1.234567") + Decimal(
            "50100.789012"
        ) * Decimal("0.987654")
        snapshot = service._market_data["BTC-USD"]
        assert snapshot.depth_l1 == expected_depth_l1
