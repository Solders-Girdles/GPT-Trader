"""Tests for SimulatedBroker core: connection, account, and initialization."""

from datetime import datetime
from decimal import Decimal

import pytest

from gpt_trader.backtesting.simulation.broker import SimulatedBroker
from gpt_trader.backtesting.types import FeeTier
from gpt_trader.core import MarketType, Position, Product


class TestSimulatedBrokerConnection:
    """Test connection lifecycle methods."""

    def test_initial_connection_state(self) -> None:
        broker = SimulatedBroker()
        assert broker.connected is False
        assert broker.validate_connection() is False

    def test_connect(self) -> None:
        broker = SimulatedBroker()
        result = broker.connect()
        assert result is True
        assert broker.connected is True
        assert broker.validate_connection() is True

    def test_disconnect(self) -> None:
        broker = SimulatedBroker()
        broker.connect()
        broker.disconnect()
        assert broker.connected is False
        assert broker.validate_connection() is False

    def test_connect_disconnect_cycle(self) -> None:
        broker = SimulatedBroker()
        broker.connect()
        assert broker.validate_connection() is True
        broker.disconnect()
        assert broker.validate_connection() is False
        broker.connect()
        assert broker.validate_connection() is True

    def test_get_account_id(self) -> None:
        broker = SimulatedBroker()
        assert broker.get_account_id() == "SIMULATED_ACCOUNT"
        assert broker.get_account_id() == broker.get_account_id()


class TestSimulatedBrokerInitialization:
    """Test SimulatedBroker initialization."""

    def test_default_initialization(self) -> None:
        broker = SimulatedBroker()
        assert broker.equity == Decimal("100000")
        assert broker.fee_tier == FeeTier.TIER_2
        assert broker.connected is False
        assert len(broker.products) == 0
        assert len(broker.positions) == 0

    def test_custom_equity_initialization(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("50000"))
        assert broker.equity == Decimal("50000")
        assert broker.get_equity() == Decimal("50000")

    def test_custom_fee_tier_initialization(self) -> None:
        broker = SimulatedBroker(fee_tier=FeeTier.TIER_5)
        assert broker.fee_tier == FeeTier.TIER_5

    def test_initial_balance_matches_equity(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("75000"))
        balances = broker.list_balances()
        assert len(balances) == 1
        assert balances[0].asset == "USDC"
        assert balances[0].total == Decimal("75000")
        assert balances[0].available == Decimal("75000")


class TestSimulatedBrokerAccountInfo:
    """Test account information methods."""

    def test_get_equity(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("123456.78"))
        assert broker.get_equity() == Decimal("123456.78")

    def test_get_account_info(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("50000"))
        info = broker.get_account_info()
        assert "cash" in info
        assert info["cash"] == Decimal("50000")

    def test_list_balances_structure(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))
        balances = broker.list_balances()
        assert len(balances) == 1
        balance = balances[0]
        assert hasattr(balance, "asset")
        assert hasattr(balance, "total")
        assert hasattr(balance, "available")

    def test_get_account_info_all_fields(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))
        info = broker.get_account_info()
        assert "cash" in info
        assert "equity" in info
        assert "unrealized_pnl" in info
        assert "realized_pnl" in info
        assert "margin_used" in info

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


class TestSimulatedBrokerPositions:
    """Test position management."""

    def test_list_positions_initially_empty(self) -> None:
        broker = SimulatedBroker()
        positions = broker.list_positions()
        assert positions == []
        assert len(positions) == 0

    def test_positions_dict_initially_empty(self) -> None:
        broker = SimulatedBroker()
        assert broker.positions == {}


class TestSimulatedBrokerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_initial_equity(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("0"))
        assert broker.equity == Decimal("0")
        assert broker.get_equity() == Decimal("0")
        balances = broker.list_balances()
        assert balances[0].total == Decimal("0")

    def test_very_large_equity(self) -> None:
        large_equity = Decimal("1000000000000")
        broker = SimulatedBroker(initial_equity_usd=large_equity)
        assert broker.equity == large_equity
        assert broker.get_equity() == large_equity

    def test_fractional_equity(self) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("12345.6789"))
        assert broker.equity == Decimal("12345.6789")


class TestSimulatedBrokerProductManagement:
    """Test product registration and retrieval."""

    @pytest.fixture
    def sample_product(self) -> Product:
        return Product(
            symbol="BTC-PERP-USDC",
            base_asset="BTC",
            quote_asset="USDC",
            market_type=MarketType.PERPETUAL,
            min_size=Decimal("0.0001"),
            step_size=Decimal("0.0001"),
            min_notional=Decimal("10"),
            price_increment=Decimal("0.01"),
            leverage_max=10,
        )

    def test_register_product(self, sample_product: Product) -> None:
        broker = SimulatedBroker()
        broker.register_product(sample_product)
        assert "BTC-PERP-USDC" in broker.products
        assert broker.products["BTC-PERP-USDC"] == sample_product

    def test_get_product_exists(self, sample_product: Product) -> None:
        broker = SimulatedBroker()
        broker.register_product(sample_product)
        retrieved = broker.get_product("BTC-PERP-USDC")
        assert retrieved is not None
        assert retrieved.symbol == "BTC-PERP-USDC"
        assert retrieved.base_asset == "BTC"

    def test_get_product_not_exists(self) -> None:
        broker = SimulatedBroker()
        retrieved = broker.get_product("NONEXISTENT-USD")
        assert retrieved is None


class TestSimulatedBrokerEquityTracking:
    """Test equity tracking and drawdown calculation."""

    def test_equity_curve_storage(self) -> None:
        broker = SimulatedBroker()
        assert broker._equity_curve == []
        broker._equity_curve = [
            (datetime(2024, 1, 1), Decimal("100000")),
            (datetime(2024, 1, 2), Decimal("101000")),
        ]
        curve = broker.get_equity_curve()
        assert len(curve) == 2
        assert curve[0][1] == Decimal("100000")

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ("_peak_equity", Decimal("75000")),
            ("_max_drawdown", Decimal("0")),
            ("_max_drawdown_usd", Decimal("0")),
        ],
    )
    def test_equity_tracking_defaults(self, attr: str, expected: Decimal) -> None:
        broker = SimulatedBroker(initial_equity_usd=Decimal("75000"))
        assert getattr(broker, attr) == expected


class TestSimulatedBrokerStatistics:
    """Test statistics tracking."""

    @pytest.mark.parametrize(
        "attr, expected",
        [
            ("_total_fees_paid", Decimal("0")),
            ("_total_trades", 0),
            ("_winning_trades", 0),
            ("_losing_trades", 0),
            ("_total_slippage_bps", Decimal("0")),
        ],
    )
    def test_statistics_defaults(self, attr: str, expected: int | Decimal) -> None:
        assert getattr(SimulatedBroker(), attr) == expected
