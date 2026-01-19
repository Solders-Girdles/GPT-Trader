"""Tests for `Ticker` in `coinbase/market_data_service.py`."""

from __future__ import annotations

from gpt_trader.features.brokerages.coinbase.market_data_service import Ticker
from gpt_trader.utilities.datetime_helpers import utc_now


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
        ticker = Ticker(
            symbol="TEST",
            bid=0.0,
            ask=0.0,
            last=0.0,
            ts=utc_now(),
        )

        assert ticker.bid == 0.0
        assert ticker.ask == 0.0
        assert ticker.last == 0.0

    def test_ticker_with_negative_values(self) -> None:
        """Test ticker with negative values (edge case, shouldn't happen in practice)."""
        ticker = Ticker(
            symbol="TEST",
            bid=-1.0,
            ask=-0.5,
            last=-0.75,
            ts=utc_now(),
        )

        assert ticker.bid == -1.0
