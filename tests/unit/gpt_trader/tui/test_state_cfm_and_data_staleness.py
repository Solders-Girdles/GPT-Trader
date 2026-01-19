"""
Tests for CFM balance state and data staleness helpers.
"""

from decimal import Decimal

from gpt_trader.core.account import CFMBalance
from gpt_trader.tui.state import TuiState


class TestTuiStateCfmBalance:
    def test_initial_cfm_state(self) -> None:
        """Test that CFM state starts as unavailable."""
        state = TuiState()
        assert state.has_cfm_access is False
        assert state.cfm_balance is None

    def test_update_cfm_balance_with_data(self) -> None:
        """Test updating CFM balance sets has_cfm_access to True."""
        state = TuiState()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("150.50"),
            daily_realized_pnl=Decimal("75.25"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )

        state.update_cfm_balance(cfm_balance)

        assert state.has_cfm_access is True
        assert state.cfm_balance is not None
        assert state.cfm_balance.futures_buying_power == Decimal("10000.00")
        assert state.cfm_balance.liquidation_buffer_percentage == 90.0

    def test_update_cfm_balance_with_none(self) -> None:
        """Test updating CFM balance to None sets has_cfm_access to False."""
        state = TuiState()

        cfm_balance = CFMBalance(
            futures_buying_power=Decimal("10000.00"),
            total_usd_balance=Decimal("5000.00"),
            available_margin=Decimal("4000.00"),
            initial_margin=Decimal("1000.00"),
            unrealized_pnl=Decimal("0"),
            daily_realized_pnl=Decimal("0"),
            liquidation_threshold=Decimal("500.00"),
            liquidation_buffer_amount=Decimal("4500.00"),
            liquidation_buffer_percentage=90.0,
        )
        state.update_cfm_balance(cfm_balance)
        assert state.has_cfm_access is True

        state.update_cfm_balance(None)
        assert state.has_cfm_access is False
        assert state.cfm_balance is None


class TestTuiStateDataStaleness:
    def test_initial_data_state_properties(self) -> None:
        """Test that data state properties start with correct defaults."""
        state = TuiState()
        assert state.data_fetching is False
        assert state.data_available is False
        assert state.last_data_fetch == 0.0

    def test_data_state_properties_can_be_set(self) -> None:
        """Test that data state properties can be set."""
        import time

        state = TuiState()

        state.data_fetching = True
        assert state.data_fetching is True

        state.data_available = True
        assert state.data_available is True

        now = time.time()
        state.last_data_fetch = now
        assert state.last_data_fetch == now

    def test_is_data_stale_false_when_no_data(self) -> None:
        """Test is_data_stale returns False when no data has been received."""
        state = TuiState()
        assert state.is_data_stale is False

        state.data_available = True
        assert state.is_data_stale is False

    def test_is_data_stale_false_when_data_fresh(self) -> None:
        """Test is_data_stale returns False when data is recent."""
        import time

        state = TuiState()
        state.data_available = True
        state.last_data_fetch = time.time()

        assert state.is_data_stale is False

    def test_is_data_stale_true_when_data_old(self) -> None:
        """Test is_data_stale returns True when data is older than 30s."""
        import time

        state = TuiState()
        state.data_available = True
        state.last_data_fetch = time.time() - 35

        assert state.is_data_stale is True

    def test_is_data_stale_false_at_threshold(self) -> None:
        """Test is_data_stale returns False when data is just under threshold."""
        import time

        state = TuiState()
        state.data_available = True
        state.last_data_fetch = time.time() - 29

        assert state.is_data_stale is False
