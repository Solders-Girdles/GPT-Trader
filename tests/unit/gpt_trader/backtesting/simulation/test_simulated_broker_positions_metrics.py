"""Tests for SimulatedBroker position risk and PnL helpers."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class TestSimulatedBrokerPositionPnl:
    """Tests for get_position_pnl method."""

    def test_pnl_with_position(self) -> None:
        from gpt_trader.core import Position

        broker = SimulatedBroker()
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("55000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("1000"),
            leverage=5,
        )

        pnl = broker.get_position_pnl("BTC-USD")

        assert pnl["realized_pnl"] == Decimal("1000")
        assert pnl["unrealized_pnl"] == Decimal("5000")
        assert pnl["total_pnl"] == Decimal("6000")

    def test_pnl_no_position(self) -> None:
        broker = SimulatedBroker()

        pnl = broker.get_position_pnl("BTC-USD")

        assert pnl["realized_pnl"] == Decimal("0")
        assert pnl["unrealized_pnl"] == Decimal("0")
        assert pnl["total_pnl"] == Decimal("0")


class TestSimulatedBrokerPositionRisk:
    """Tests for get_position_risk method."""

    def test_risk_with_long_position(self) -> None:
        from datetime import datetime

        from gpt_trader.core import Position, Quote

        broker = SimulatedBroker()
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        # Add quote for mark price
        broker._current_quote["BTC-USD"] = Quote(
            symbol="BTC-USD",
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            last=Decimal("50000"),
            ts=datetime(2024, 1, 1, 12, 0, 0),
        )

        risk = broker.get_position_risk("BTC-USD")

        assert risk["symbol"] == "BTC-USD"
        assert risk["notional"] == Decimal("50000")
        assert risk["leverage"] == 5
        assert risk["margin_used"] == Decimal("10000")
        assert risk["entry_price"] == Decimal("50000")
        assert risk["mark_price"] == Decimal("50000")
        assert "liquidation_price" in risk

    def test_risk_no_position(self) -> None:
        broker = SimulatedBroker()

        risk = broker.get_position_risk("BTC-USD")

        assert risk == {}
