"""Tests for SimulatedBroker connection lifecycle and basic account state."""

from decimal import Decimal

from gpt_trader.backtesting.simulation.broker import SimulatedBroker


class TestSimulatedBrokerConnection:
    """Test connection lifecycle methods."""

    def test_initial_connection_state(self) -> None:
        """Test broker starts disconnected."""
        broker = SimulatedBroker()

        assert broker.connected is False
        assert broker.validate_connection() is False

    def test_connect(self) -> None:
        """Test connect() sets connected state."""
        broker = SimulatedBroker()

        result = broker.connect()

        assert result is True
        assert broker.connected is True
        assert broker.validate_connection() is True

    def test_disconnect(self) -> None:
        """Test disconnect() clears connected state."""
        broker = SimulatedBroker()
        broker.connect()

        broker.disconnect()

        assert broker.connected is False
        assert broker.validate_connection() is False

    def test_connect_disconnect_cycle(self) -> None:
        """Test multiple connect/disconnect cycles."""
        broker = SimulatedBroker()

        # First cycle
        broker.connect()
        assert broker.validate_connection() is True
        broker.disconnect()
        assert broker.validate_connection() is False

        # Second cycle
        broker.connect()
        assert broker.validate_connection() is True

    def test_get_account_id(self) -> None:
        """Test get_account_id returns fixed value."""
        broker = SimulatedBroker()

        assert broker.get_account_id() == "SIMULATED_ACCOUNT"

    def test_get_account_id_consistent(self) -> None:
        """Test get_account_id is consistent across calls."""
        broker = SimulatedBroker()

        id1 = broker.get_account_id()
        id2 = broker.get_account_id()

        assert id1 == id2


class TestSimulatedBrokerAccountInfo:
    """Test account information methods."""

    def test_get_equity(self) -> None:
        """Test get_equity returns initial equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("123456.78"))

        assert broker.get_equity() == Decimal("123456.78")

    def test_get_account_info(self) -> None:
        """Test get_account_info returns cash key."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("50000"))

        info = broker.get_account_info()

        assert "cash" in info
        assert info["cash"] == Decimal("50000")

    def test_list_balances_structure(self) -> None:
        """Test list_balances returns proper Balance objects."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("100000"))

        balances = broker.list_balances()

        assert len(balances) == 1
        balance = balances[0]
        assert hasattr(balance, "asset")
        assert hasattr(balance, "total")
        assert hasattr(balance, "available")


class TestSimulatedBrokerPositions:
    """Test position management."""

    def test_list_positions_initially_empty(self) -> None:
        """Test positions list is empty initially."""
        broker = SimulatedBroker()

        positions = broker.list_positions()

        assert positions == []
        assert len(positions) == 0

    def test_positions_dict_initially_empty(self) -> None:
        """Test positions dict is empty initially."""
        broker = SimulatedBroker()

        assert broker.positions == {}


class TestSimulatedBrokerEdgeCases:
    """Test edge cases and special scenarios."""

    def test_zero_initial_equity(self) -> None:
        """Test initialization with zero equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("0"))

        assert broker.equity == Decimal("0")
        assert broker.get_equity() == Decimal("0")
        balances = broker.list_balances()
        assert balances[0].total == Decimal("0")

    def test_very_large_equity(self) -> None:
        """Test initialization with very large equity."""
        large_equity = Decimal("1000000000000")  # 1 trillion
        broker = SimulatedBroker(initial_equity_usd=large_equity)

        assert broker.equity == large_equity
        assert broker.get_equity() == large_equity

    def test_fractional_equity(self) -> None:
        """Test initialization with fractional equity."""
        broker = SimulatedBroker(initial_equity_usd=Decimal("12345.6789"))

        assert broker.equity == Decimal("12345.6789")
