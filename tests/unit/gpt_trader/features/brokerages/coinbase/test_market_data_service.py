"""Tests for CoinbaseTickerService and related market data service components."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.market_data_service import (
    CoinbaseTickerService,
    MarketDataService,
    Ticker,
    TickerCache,
)
from gpt_trader.utilities.datetime_helpers import utc_now


class TestMarketDataServiceAlias:
    """Tests for MarketDataService alias."""

    def test_alias_is_coinbase_ticker_service(self) -> None:
        """Test that MarketDataService is an alias for CoinbaseTickerService."""
        assert MarketDataService is CoinbaseTickerService

    def test_alias_instantiation(self) -> None:
        """Test that MarketDataService can be instantiated."""
        service = MarketDataService(symbols=["BTC-USD"])

        assert isinstance(service, CoinbaseTickerService)
        assert service._symbols == ["BTC-USD"]


class TestTicker:
    """Tests for Ticker dataclass."""

    def test_ticker_creation(self) -> None:
        """Test creating a Ticker instance."""
        now = utc_now()
        ticker = Ticker(
            symbol="BTC-USD",
            bid=49900.0,
            ask=50100.0,
            last=50000.0,
            ts=now,
        )

        assert ticker.symbol == "BTC-USD"
        assert ticker.bid == 49900.0
        assert ticker.ask == 50100.0
        assert ticker.last == 50000.0
        assert ticker.ts == now

    def test_ticker_equality(self) -> None:
        """Test Ticker equality based on dataclass behavior."""
        ts = utc_now()
        ticker1 = Ticker(symbol="BTC-USD", bid=100.0, ask=101.0, last=100.5, ts=ts)
        ticker2 = Ticker(symbol="BTC-USD", bid=100.0, ask=101.0, last=100.5, ts=ts)

        assert ticker1 == ticker2

    def test_ticker_different_symbols(self) -> None:
        """Test Tickers with different symbols are not equal."""
        ts = utc_now()
        ticker1 = Ticker(symbol="BTC-USD", bid=100.0, ask=101.0, last=100.5, ts=ts)
        ticker2 = Ticker(symbol="ETH-USD", bid=100.0, ask=101.0, last=100.5, ts=ts)

        assert ticker1 != ticker2

    def test_ticker_with_zero_values(self) -> None:
        """Test ticker with zero bid/ask/last."""
        ticker = Ticker(symbol="TEST", bid=0.0, ask=0.0, last=0.0, ts=utc_now())

        assert ticker.bid == 0.0
        assert ticker.ask == 0.0
        assert ticker.last == 0.0

    def test_ticker_with_negative_values(self) -> None:
        """Test ticker with negative values (edge case, shouldn't happen in practice)."""
        ticker = Ticker(symbol="TEST", bid=-1.0, ask=-0.5, last=-0.75, ts=utc_now())

        assert ticker.bid == -1.0


class TestCoinbaseTickerService:
    """Tests for CoinbaseTickerService class."""

    def test_init_default_symbols(self) -> None:
        """Test initialization with default empty symbols."""
        service = CoinbaseTickerService()

        assert service._symbols == []
        assert service._running is False
        assert service._thread is None

    def test_init_with_symbols(self) -> None:
        """Test initialization with provided symbols."""
        service = CoinbaseTickerService(symbols=["BTC-USD", "ETH-USD"])

        assert service._symbols == ["BTC-USD", "ETH-USD"]

    def test_set_symbols(self) -> None:
        """Test setting symbols after initialization."""
        service = CoinbaseTickerService()
        service.set_symbols(["SOL-USD", "DOGE-USD"])

        assert service._symbols == ["SOL-USD", "DOGE-USD"]

    def test_start_sets_running(self) -> None:
        """Test that start sets running flag and creates thread."""
        service = CoinbaseTickerService()
        service.start()

        assert service._running is True
        assert service._thread is not None

        service.stop()

    def test_stop_clears_running(self) -> None:
        """Test that stop clears running flag."""
        service = CoinbaseTickerService()
        service.start()
        service.stop()

        assert service._running is False

    def test_stop_without_start(self) -> None:
        """Test that stop handles case where thread was never started."""
        service = CoinbaseTickerService()
        service.stop()

        assert service._running is False

    def test_get_mark_returns_none(self) -> None:
        """Test that get_mark returns None (stub implementation)."""
        service = CoinbaseTickerService()

        result = service.get_mark("BTC-USD")

        assert result is None

    def test_service_lifecycle(self) -> None:
        """Test full start/stop lifecycle."""
        service = CoinbaseTickerService(symbols=["BTC-USD"])

        service.start()
        assert service._running is True
        assert service._thread is not None

        service.stop()
        assert service._running is False

    def test_run_method_does_nothing(self) -> None:
        """Test that _run method is a no-op (stub)."""
        service = CoinbaseTickerService()
        service._run()

        assert service._running is False
        assert service._thread is None

    def test_service_set_symbols_replaces(self) -> None:
        """Test that set_symbols replaces all symbols."""
        service = CoinbaseTickerService(symbols=["BTC-USD", "ETH-USD"])
        service.set_symbols(["SOL-USD"])

        assert service._symbols == ["SOL-USD"]
        assert "BTC-USD" not in service._symbols

    def test_service_set_empty_symbols(self) -> None:
        """Test setting empty symbols list."""
        service = CoinbaseTickerService(symbols=["BTC-USD"])
        service.set_symbols([])

        assert service._symbols == []

    def test_get_last_ticker_timestamp_reads_from_cache(self) -> None:
        """Service should expose the latest timestamp tracked by its cache."""
        cache = TickerCache()
        latest = utc_now()
        cache.update(Ticker(symbol="BTC-USD", bid=1.0, ask=2.0, last=1.5, ts=latest))
        service = CoinbaseTickerService(symbols=["BTC-USD"], ticker_cache=cache)

        assert service.get_last_ticker_timestamp() == latest

    def test_multiple_start_stop_cycles(self) -> None:
        """Test multiple start/stop cycles."""
        service = CoinbaseTickerService()

        for _ in range(3):
            service.start()
            assert service._running is True
            service.stop()
            assert service._running is False
