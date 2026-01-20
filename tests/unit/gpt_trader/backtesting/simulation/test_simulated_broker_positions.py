"""Tests for SimulatedBroker positions: margin, risk, PnL, equity tracking."""

from datetime import datetime
from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.core import Position, Quote


class TestSimulatedBrokerMarginCalculation:
    """Test margin calculation methods."""

    def test_calculate_margin_used_no_positions(self) -> None:
        broker = SimulatedBroker()
        assert broker._calculate_margin_used() == Decimal("0")

    def test_calculate_margin_used_with_position(self) -> None:
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
        assert broker._calculate_margin_used() == Decimal("10000")

    def test_calculate_margin_used_multiple_positions(self) -> None:
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
        broker.positions["ETH-USD"] = Position(
            symbol="ETH-USD",
            quantity=Decimal("10.0"),
            side="long",
            entry_price=Decimal("3000"),
            mark_price=Decimal("3000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=3,
        )
        assert broker._calculate_margin_used() == Decimal("20000")

    def test_calculate_margin_used_default_leverage(self) -> None:
        broker = SimulatedBroker()
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("50000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
            leverage=None,  # type: ignore[arg-type]
        )
        assert broker._calculate_margin_used() == Decimal("50000")


class TestSimulatedBrokerPositionPnl:
    """Tests for get_position_pnl method."""

    def test_pnl_with_position(self) -> None:
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
        assert "liquidation_price" in risk

    def test_risk_no_position(self) -> None:
        broker = SimulatedBroker()
        assert broker.get_position_risk("BTC-USD") == {}


class TestSimulatedBrokerAccountInfoWithPositions:
    """Test account info with positions."""

    def test_get_account_info_with_unrealized_pnl(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("55000"),
            unrealized_pnl=Decimal("5000"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        info = broker.get_account_info()
        assert info["unrealized_pnl"] == Decimal("5000")
        assert info["equity"] == Decimal("105000")


class TestSimulatedBrokerEquityTracking:
    """Test equity tracking and drawdown calculation."""

    def test_equity_curve_initially_empty(self) -> None:
        assert SimulatedBroker()._equity_curve == []

    def test_get_equity_curve(self) -> None:
        broker = SimulatedBroker()
        broker._equity_curve = [
            (datetime(2024, 1, 1), Decimal("100000")),
            (datetime(2024, 1, 2), Decimal("101000")),
        ]
        curve = broker.get_equity_curve()
        assert len(curve) == 2
        assert curve[0][1] == Decimal("100000")

    def test_peak_equity_initialized(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("75000"))
        assert broker._peak_equity == Decimal("75000")

    def test_max_drawdown_initially_zero(self) -> None:
        broker = SimulatedBroker()
        assert broker._max_drawdown == Decimal("0")
        assert broker._max_drawdown_usd == Decimal("0")


class TestSimulatedBrokerStatistics:
    """Test statistics tracking."""

    def test_total_fees_initially_zero(self) -> None:
        assert SimulatedBroker()._total_fees_paid == Decimal("0")

    def test_trade_counters_initially_zero(self) -> None:
        broker = SimulatedBroker()
        assert broker._total_trades == 0
        assert broker._winning_trades == 0
        assert broker._losing_trades == 0

    def test_slippage_initially_zero(self) -> None:
        assert SimulatedBroker()._total_slippage_bps == Decimal("0")


class TestSimulatedBrokerGetPosition:
    """Test get_position method."""

    def test_get_position_no_position(self) -> None:
        assert SimulatedBroker().get_position("BTC-USD") is None

    def test_get_position_with_position(self) -> None:
        broker = SimulatedBroker()
        broker.positions["BTC-USD"] = Position(
            symbol="BTC-USD",
            quantity=Decimal("1.0"),
            side="long",
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            unrealized_pnl=Decimal("1000"),
            realized_pnl=Decimal("0"),
            leverage=5,
        )
        result = broker.get_position("BTC-USD")
        assert result is not None
        assert result.symbol == "BTC-USD"
        assert result.quantity == Decimal("1.0")
