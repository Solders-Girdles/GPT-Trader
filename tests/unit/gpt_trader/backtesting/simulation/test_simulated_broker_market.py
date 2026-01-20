"""Tests for SimulatedBroker market data: quotes, tickers, candles, mark prices."""

from datetime import datetime, timedelta
from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.core import Candle, Quote


class TestSimulatedBrokerQuote:
    """Test quote methods."""

    def test_get_quote_no_data(self) -> None:
        broker = SimulatedBroker()
        quote = broker.get_quote("BTC-USD")
        assert quote is None

    def test_get_quote_with_data(self) -> None:
        broker = SimulatedBroker()
        test_quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = test_quote

        result = broker.get_quote("BTC-USD")

        assert result is not None
        assert result.symbol == "BTC-USD"
        assert result.bid == Decimal("49990")
        assert result.ask == Decimal("50010")


class TestSimulatedBrokerTicker:
    """Test ticker methods."""

    def test_get_ticker_with_quote(self) -> None:
        broker = SimulatedBroker()
        test_quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = test_quote

        ticker = broker.get_ticker("BTC-USD")

        assert ticker["price"] == "50000"
        assert ticker["bid"] == "49990"
        assert ticker["ask"] == "50010"

    def test_get_ticker_with_bar(self) -> None:
        broker = SimulatedBroker()
        test_bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = test_bar

        ticker = broker.get_ticker("BTC-USD")

        assert ticker["price"] == "50500"
        assert ticker["volume"] == "100"

    def test_get_ticker_no_data(self) -> None:
        broker = SimulatedBroker()
        ticker = broker.get_ticker("BTC-USD")
        assert ticker["price"] == "0"
        assert ticker["bid"] == "0"
        assert ticker["ask"] == "0"
        assert ticker["volume"] == "0"


class TestSimulatedBrokerCandles:
    """Test candle methods."""

    def test_get_candles_empty(self) -> None:
        broker = SimulatedBroker()
        candles = broker.get_candles("BTC-USD")
        assert candles == []

    def test_get_candles_with_history(self) -> None:
        broker = SimulatedBroker()
        test_candles = [
            Candle(
                ts=datetime(2024, 1, 1, i, 0, 0),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("100"),
            )
            for i in range(5)
        ]
        broker._candle_history["BTC-USD"] = test_candles

        result = broker.get_candles("BTC-USD")

        assert len(result) == 5

    def test_get_candles_with_limit(self) -> None:
        broker = SimulatedBroker()
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        test_candles = [
            Candle(
                ts=base_time + timedelta(hours=i),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("100"),
            )
            for i in range(100)
        ]
        broker._candle_history["BTC-USD"] = test_candles

        result = broker.get_candles("BTC-USD", limit=10)

        assert len(result) == 10


class TestSimulatedBrokerMarkPrice:
    """Tests for get_mark_price method."""

    def test_get_mark_price_from_quote(self) -> None:
        broker = SimulatedBroker()
        quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = quote

        mark = broker.get_mark_price("BTC-USD")

        assert mark == Decimal("50000")

    def test_get_mark_price_from_bar(self) -> None:
        broker = SimulatedBroker()
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        mark = broker.get_mark_price("BTC-USD")

        assert mark == Decimal("50500")

    def test_get_mark_price_no_data(self) -> None:
        broker = SimulatedBroker()
        mark = broker.get_mark_price("BTC-USD")
        assert mark is None


class TestSimulatedBrokerMarketSnapshot:
    """Tests for get_market_snapshot method."""

    def test_snapshot_with_quote(self) -> None:
        broker = SimulatedBroker()
        quote = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )
        broker._current_quote["BTC-USD"] = quote

        snapshot = broker.get_market_snapshot("BTC-USD")

        assert snapshot["symbol"] == "BTC-USD"
        assert snapshot["bid"] == Decimal("49990")
        assert snapshot["ask"] == Decimal("50010")
        assert snapshot["last"] == Decimal("50000")
        assert snapshot["spread"] == Decimal("20")
        assert snapshot["mid"] == Decimal("50000")

    def test_snapshot_with_bar(self) -> None:
        broker = SimulatedBroker()
        bar = Candle(
            ts=datetime(2024, 1, 1, 12, 0, 0),
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48500"),
            close=Decimal("50500"),
            volume=Decimal("100"),
        )
        broker._current_bar["BTC-USD"] = bar

        snapshot = broker.get_market_snapshot("BTC-USD")

        assert snapshot["symbol"] == "BTC-USD"
        assert snapshot["open"] == Decimal("49000")
        assert snapshot["high"] == Decimal("51000")
        assert snapshot["low"] == Decimal("48500")
        assert snapshot["close"] == Decimal("50500")
        assert snapshot["volume"] == Decimal("100")

    def test_snapshot_empty(self) -> None:
        broker = SimulatedBroker()
        snapshot = broker.get_market_snapshot("BTC-USD")
        assert snapshot == {"symbol": "BTC-USD"}
