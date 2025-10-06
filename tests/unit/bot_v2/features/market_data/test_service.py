"""Unit tests for MarketDataService.

Tests cover:
- Success path (quote fetch, mark update, timestamp update)
- Error handling (one symbol failure doesn't block others)
- Window trimming
- Thread safety (lock usage)
- Risk manager timestamp updates
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from bot_v2.features.market_data import MarketDataService


@pytest.fixture
def mock_broker():
    """Mock broker with get_quote method."""
    broker = Mock()
    broker.get_quote = Mock()
    return broker


@pytest.fixture
def mock_risk_manager():
    """Mock risk manager with last_mark_update dict."""
    risk_manager = Mock()
    risk_manager.last_mark_update = {}
    return risk_manager


@pytest.fixture
def mock_quote():
    """Mock quote with price and timestamp."""
    quote = Mock()
    quote.last = Decimal("50000.0")
    quote.ts = datetime(2025, 10, 1, 12, 0, 0, tzinfo=UTC)
    return quote


@pytest.fixture
def market_data_service(mock_broker, mock_risk_manager):
    """Create MarketDataService with default config."""
    symbols = ["BTC-USD", "ETH-USD"]
    mark_lock = threading.RLock()
    return MarketDataService(
        symbols=symbols,
        broker=mock_broker,
        risk_manager=mock_risk_manager,
        long_ma=50,
        short_ma=20,
        mark_lock=mark_lock,
    )


class TestMarketDataServiceInitialization:
    """Test service initialization."""

    def test_initialization_creates_mark_windows(self, market_data_service):
        """Verify mark_windows initialized for all symbols."""
        assert "BTC-USD" in market_data_service.mark_windows
        assert "ETH-USD" in market_data_service.mark_windows
        assert market_data_service.mark_windows["BTC-USD"] == []
        assert market_data_service.mark_windows["ETH-USD"] == []

    def test_initialization_accepts_existing_mark_windows(self, mock_broker, mock_risk_manager):
        """Verify existing mark_windows can be passed in."""
        existing_windows = {"BTC-USD": [Decimal("49000.0")], "ETH-USD": []}
        service = MarketDataService(
            symbols=["BTC-USD", "ETH-USD"],
            broker=mock_broker,
            risk_manager=mock_risk_manager,
            long_ma=50,
            short_ma=20,
            mark_lock=threading.RLock(),
            mark_windows=existing_windows,
        )
        assert service.mark_windows is existing_windows
        assert service.mark_windows["BTC-USD"] == [Decimal("49000.0")]

    def test_mark_windows_property_exposes_internal_dict(self, market_data_service):
        """Verify mark_windows property returns internal dict."""
        assert market_data_service.mark_windows is market_data_service._mark_windows


class TestMarketDataServiceUpdateMarks:
    """Test update_marks method."""

    @pytest.mark.asyncio
    async def test_update_marks_updates_mark_windows(self, market_data_service, mock_quote):
        """Verify update_marks appends to mark_windows."""
        market_data_service.broker.get_quote.return_value = mock_quote

        await market_data_service.update_marks()

        assert len(market_data_service.mark_windows["BTC-USD"]) == 1
        assert market_data_service.mark_windows["BTC-USD"][0] == Decimal("50000.0")
        assert len(market_data_service.mark_windows["ETH-USD"]) == 1
        assert market_data_service.mark_windows["ETH-USD"][0] == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_update_marks_updates_risk_manager_timestamp(
        self, market_data_service, mock_quote
    ):
        """Verify update_marks updates risk_manager.last_mark_update."""
        market_data_service.broker.get_quote.return_value = mock_quote

        await market_data_service.update_marks()

        assert "BTC-USD" in market_data_service.risk_manager.last_mark_update
        assert "ETH-USD" in market_data_service.risk_manager.last_mark_update
        assert isinstance(market_data_service.risk_manager.last_mark_update["BTC-USD"], datetime)

    @pytest.mark.asyncio
    async def test_update_marks_continues_after_symbol_error(self, market_data_service, mock_quote):
        """Verify error on one symbol doesn't block others."""

        # BTC-USD fails, ETH-USD succeeds
        def get_quote_side_effect(symbol):
            if symbol == "BTC-USD":
                return None  # Will raise RuntimeError
            return mock_quote

        market_data_service.broker.get_quote.side_effect = get_quote_side_effect

        await market_data_service.update_marks()

        # BTC-USD should have no marks (failed)
        assert len(market_data_service.mark_windows["BTC-USD"]) == 0
        # ETH-USD should have mark (succeeded)
        assert len(market_data_service.mark_windows["ETH-USD"]) == 1
        assert market_data_service.mark_windows["ETH-USD"][0] == Decimal("50000.0")

    @pytest.mark.asyncio
    async def test_update_marks_handles_quote_without_last_price(self, market_data_service):
        """Verify error when quote has no price field."""
        bad_quote = Mock()
        bad_quote.last = None
        bad_quote.last_price = None
        market_data_service.broker.get_quote.return_value = bad_quote

        await market_data_service.update_marks()

        # Should log error and not update mark_windows
        assert len(market_data_service.mark_windows["BTC-USD"]) == 0
        assert len(market_data_service.mark_windows["ETH-USD"]) == 0

    @pytest.mark.asyncio
    async def test_update_marks_uses_last_price_fallback(self, market_data_service):
        """Verify fallback to last_price attribute if last doesn't exist."""
        quote = Mock(spec=[])  # No attributes defined
        quote.last_price = Decimal("51000.0")
        quote.ts = datetime.now(UTC)
        market_data_service.broker.get_quote.return_value = quote

        await market_data_service.update_marks()

        assert market_data_service.mark_windows["BTC-USD"][0] == Decimal("51000.0")

    @pytest.mark.asyncio
    async def test_update_marks_rejects_invalid_price(self, market_data_service):
        """Verify error when price is invalid (<=0)."""
        bad_quote = Mock()
        bad_quote.last = Decimal("0")
        market_data_service.broker.get_quote.return_value = bad_quote

        await market_data_service.update_marks()

        assert len(market_data_service.mark_windows["BTC-USD"]) == 0


class TestMarketDataServiceWindowTrimming:
    """Test window trimming behavior."""

    @pytest.mark.asyncio
    async def test_update_mark_window_trims_correctly(self, market_data_service, mock_quote):
        """Verify window trimmed to max(long_ma, short_ma) + 5."""
        market_data_service.broker.get_quote.return_value = mock_quote

        # long_ma=50, short_ma=20, so max_size = 50 + 5 = 55
        # Add 60 marks to trigger trimming
        for _ in range(60):
            await market_data_service.update_marks()

        # Should be trimmed to 55
        assert len(market_data_service.mark_windows["BTC-USD"]) == 55
        assert len(market_data_service.mark_windows["ETH-USD"]) == 55

    @pytest.mark.asyncio
    async def test_window_trimming_keeps_latest_values(self, market_data_service):
        """Verify trimming keeps most recent values."""
        # Add marks with increasing values
        for i in range(60):
            quote = Mock()
            quote.last = Decimal(str(50000 + i))
            quote.ts = datetime.now(UTC)
            market_data_service.broker.get_quote.return_value = quote
            await market_data_service.update_marks()

        # Should keep last 55 values (50005 to 50059)
        btc_marks = market_data_service.mark_windows["BTC-USD"]
        assert len(btc_marks) == 55
        assert btc_marks[0] == Decimal("50005")  # First of kept values
        assert btc_marks[-1] == Decimal("50059")  # Last value


class TestMarketDataServiceThreadSafety:
    """Test thread safety via lock usage."""

    def test_update_mark_window_uses_lock(self, monkeypatch):
        """Verify _update_mark_window uses the shared lock."""
        import threading

        # Track lock usage by patching threading.RLock class
        acquire_count = 0
        original_rlock = threading.RLock

        class InstrumentedRLock:
            def __init__(self):
                self._lock = original_rlock()

            def acquire(self, *args, **kwargs):
                nonlocal acquire_count
                acquire_count += 1
                return self._lock.acquire(*args, **kwargs)

            def release(self):
                return self._lock.release()

            def __enter__(self):
                self.acquire()
                return self

            def __exit__(self, *args):
                self.release()

        monkeypatch.setattr(threading, "RLock", InstrumentedRLock)

        # Create service with instrumented lock
        from bot_v2.features.market_data import MarketDataService

        broker = Mock()
        risk_manager = Mock()
        risk_manager.last_mark_update = {}

        service = MarketDataService(
            symbols=["BTC-USD"],
            broker=broker,
            risk_manager=risk_manager,
            long_ma=50,
            short_ma=20,
            mark_lock=threading.RLock(),
        )

        # Call _update_mark_window
        service._update_mark_window("BTC-USD", Decimal("50000.0"))

        # Verify lock was acquired
        assert acquire_count >= 1

    @pytest.mark.asyncio
    async def test_concurrent_updates_are_safe(self, market_data_service, mock_quote):
        """Verify concurrent update_marks calls don't corrupt mark_windows."""
        import asyncio

        market_data_service.broker.get_quote.return_value = mock_quote

        # Run 10 concurrent updates
        await asyncio.gather(*[market_data_service.update_marks() for _ in range(10)])

        # Should have 10 marks per symbol (2 symbols)
        assert len(market_data_service.mark_windows["BTC-USD"]) == 10
        assert len(market_data_service.mark_windows["ETH-USD"]) == 10

        # All marks should be valid Decimals
        for mark in market_data_service.mark_windows["BTC-USD"]:
            assert isinstance(mark, Decimal)
            assert mark == Decimal("50000.0")


class TestMarketDataServiceRiskManagerIntegration:
    """Test risk manager timestamp updates."""

    @pytest.mark.asyncio
    async def test_risk_manager_timestamp_uses_quote_ts(self, market_data_service):
        """Verify risk manager gets timestamp from quote.ts."""
        quote = Mock()
        quote.last = Decimal("50000.0")
        quote.ts = datetime(2025, 10, 1, 15, 30, 0, tzinfo=UTC)
        market_data_service.broker.get_quote.return_value = quote

        await market_data_service.update_marks()

        assert market_data_service.risk_manager.last_mark_update["BTC-USD"] == datetime(
            2025, 10, 1, 15, 30, 0, tzinfo=UTC
        )

    @pytest.mark.asyncio
    async def test_risk_manager_timestamp_continues_on_error(self, market_data_service, mock_quote):
        """Verify timestamp update errors don't crash update_marks."""
        market_data_service.broker.get_quote.return_value = mock_quote

        # Make risk manager raise on attribute access
        class FailingDict(dict):
            def __setitem__(self, key, value):
                raise RuntimeError("Timestamp update failed")

        market_data_service.risk_manager.last_mark_update = FailingDict()

        # Should not raise, just log debug
        await market_data_service.update_marks()

        # Mark windows should still be updated
        assert len(market_data_service.mark_windows["BTC-USD"]) == 1
