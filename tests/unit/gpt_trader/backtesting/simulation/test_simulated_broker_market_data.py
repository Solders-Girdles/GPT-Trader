"""Tests for SimulatedBroker market data helpers."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class TestSimulatedBrokerMarketData:
    """Test market data methods."""

    def test_get_quote_no_data(self) -> None:
        """Test get_quote returns None when no data available."""
        broker = SimulatedBroker()

        quote = broker.get_quote("BTC-USD")

        assert quote is None

    def test_get_quote_with_data(self) -> None:
        """Test get_quote returns stored quote."""
        from datetime import datetime

        from gpt_trader.core import Quote

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

    def test_get_position_no_position(self) -> None:
        """Test get_position returns None when no position exists."""
        broker = SimulatedBroker()

        position = broker.get_position("BTC-USD")

        assert position is None

    def test_get_position_with_position(self) -> None:
        """Test get_position returns stored position."""
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        test_position = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("1000"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        broker.positions["BTC-USD"] = test_position

        result = broker.get_position("BTC-USD")

        assert result is not None
        assert result.symbol == "BTC-USD"
        assert result.quantity == Decimal("1.0")

    def test_get_ticker_with_quote(self) -> None:
        """Test get_ticker returns data from quote."""
        from datetime import datetime

        from gpt_trader.core import Quote

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
        """Test get_ticker returns data from bar when no quote."""
        from datetime import datetime

        from gpt_trader.core import Candle

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
        """Test get_ticker returns zeros when no data."""
        broker = SimulatedBroker()

        ticker = broker.get_ticker("BTC-USD")

        assert ticker["price"] == "0"
        assert ticker["bid"] == "0"
        assert ticker["ask"] == "0"
        assert ticker["volume"] == "0"

    def test_get_candles_empty(self) -> None:
        """Test get_candles returns empty list when no history."""
        broker = SimulatedBroker()

        candles = broker.get_candles("BTC-USD")

        assert candles == []

    def test_get_candles_with_history(self) -> None:
        """Test get_candles returns stored candles."""
        from datetime import datetime

        from gpt_trader.core import Candle

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
        """Test get_candles respects limit parameter."""
        from datetime import datetime, timedelta

        from gpt_trader.core import Candle

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
