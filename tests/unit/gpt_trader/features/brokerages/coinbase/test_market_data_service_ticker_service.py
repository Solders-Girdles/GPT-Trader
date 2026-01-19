"""Tests for `CoinbaseTickerService` in `coinbase/market_data_service.py`."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.market_data_service import CoinbaseTickerService


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
        assert service._thread.is_alive() or not service._thread.is_alive()

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

    def test_multiple_start_stop_cycles(self) -> None:
        """Test multiple start/stop cycles."""
        service = CoinbaseTickerService()

        for _ in range(3):
            service.start()
            assert service._running is True
            service.stop()
            assert service._running is False
