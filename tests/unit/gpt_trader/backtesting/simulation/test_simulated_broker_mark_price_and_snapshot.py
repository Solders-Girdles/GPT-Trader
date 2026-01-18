"""Tests for SimulatedBroker mark price and market snapshot helpers."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class TestSimulatedBrokerMarkPrice:
    """Tests for get_mark_price method."""

    def test_get_mark_price_from_quote(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Quote

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
        from datetime import datetime

        from gpt_trader.core import Candle

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
        from datetime import datetime

        from gpt_trader.core import Quote

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
        from datetime import datetime

        from gpt_trader.core import Candle

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
