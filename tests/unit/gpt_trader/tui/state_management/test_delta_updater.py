"""Tests for StateDeltaUpdater comparisons."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.tui.state_management.delta_updater import DeltaResult, StateDeltaUpdater
from gpt_trader.tui.types import (
    AccountBalance,
    AccountSummary,
    RiskGuard,
    RiskState,
    SystemStatus,
)


class TestDeltaResult:
    """Test DeltaResult dataclass."""

    def test_initial_state_no_changes(self):
        """Test new DeltaResult has no changes."""
        result = DeltaResult()

        assert result.has_changes is False
        assert result.changed_fields == []
        assert result.details == {}

    def test_add_change_sets_has_changes(self):
        """Test adding a change sets has_changes flag."""
        result = DeltaResult()
        result.add_change("field", "old", "new")

        assert result.has_changes is True
        assert "field" in result.changed_fields
        assert result.details["field"] == ("old", "new")

    def test_add_multiple_changes(self):
        """Test adding multiple changes."""
        result = DeltaResult()
        result.add_change("field1", "old1", "new1")
        result.add_change("field2", "old2", "new2")

        assert len(result.changed_fields) == 2
        assert len(result.details) == 2


class TestStateDeltaUpdaterHelpers:
    """Test StateDeltaUpdater helper methods."""

    def test_should_update_component_no_changes(self):
        """Test should_update returns False when no changes."""
        updater = StateDeltaUpdater()
        delta = DeltaResult()

        assert not updater.should_update_component("market", delta)

    def test_should_update_component_with_changes(self):
        """Test should_update returns True when there are changes."""
        updater = StateDeltaUpdater()
        delta = DeltaResult()
        delta.add_change("prices.BTC-USD", "50000", "51000")

        assert updater.should_update_component("market", delta)

    def test_float_equal_within_epsilon(self):
        """Test float comparison with epsilon tolerance."""
        updater = StateDeltaUpdater()

        assert updater._float_equal(1.0, 1.0)
        assert updater._float_equal(1.0, 1.0 + 1e-10)  # Within epsilon
        assert not updater._float_equal(1.0, 1.1)  # Beyond epsilon

    def test_decimal_equal_within_epsilon(self):
        """Test decimal comparison with epsilon tolerance."""
        updater = StateDeltaUpdater()

        assert updater._decimal_equal(Decimal("1.0"), Decimal("1.0"))
        assert updater._decimal_equal(Decimal("1.0"), Decimal("1.000000001"))
        assert not updater._decimal_equal(Decimal("1.0"), Decimal("1.1"))

    def test_decimal_equal_handles_none(self):
        """Test decimal comparison handles None values."""
        updater = StateDeltaUpdater()

        assert updater._decimal_equal(None, None)
        assert not updater._decimal_equal(Decimal("1.0"), None)
        assert not updater._decimal_equal(None, Decimal("1.0"))


class TestStateDeltaUpdaterAccountRiskSystem:
    """Test StateDeltaUpdater comparisons for account, risk, and system."""

    def test_compare_account_no_changes(self):
        """Test account comparison with identical data."""
        updater = StateDeltaUpdater()
        account = AccountSummary(
            volume_30d=Decimal("100000"),
            fees_30d=Decimal("500"),
            fee_tier="Advanced",
            balances=[
                AccountBalance(
                    asset="USD",
                    total=Decimal("10000"),
                    available=Decimal("9000"),
                )
            ],
        )

        result = updater.compare_account(account, account)

        assert not result.has_changes

    def test_compare_account_balance_change(self):
        """Test account comparison detects balance change."""
        updater = StateDeltaUpdater()
        old_account = AccountSummary(
            volume_30d=Decimal("100000"),
            fees_30d=Decimal("500"),
            fee_tier="Advanced",
            balances=[
                AccountBalance(
                    asset="USD",
                    total=Decimal("10000"),
                    available=Decimal("9000"),
                )
            ],
        )
        new_account = AccountSummary(
            volume_30d=Decimal("100000"),
            fees_30d=Decimal("500"),
            fee_tier="Advanced",
            balances=[
                AccountBalance(
                    asset="USD",
                    total=Decimal("11000"),
                    available=Decimal("10000"),
                )
            ],
        )

        result = updater.compare_account(old_account, new_account)

        assert result.has_changes
        assert "balances.USD.total" in result.changed_fields

    def test_compare_risk_no_changes(self):
        """Test risk comparison with identical data."""
        updater = StateDeltaUpdater()
        risk = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=5.0,
            current_daily_loss_pct=1.0,
            reduce_only_mode=False,
            reduce_only_reason="",
            guards=[RiskGuard(name="guard1")],
        )

        result = updater.compare_risk(risk, risk)

        assert not result.has_changes

    def test_compare_risk_reduce_only_change(self):
        """Test risk comparison detects reduce_only mode change."""
        updater = StateDeltaUpdater()
        old_risk = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=5.0,
            current_daily_loss_pct=1.0,
            reduce_only_mode=False,
            reduce_only_reason="",
        )
        new_risk = RiskState(
            max_leverage=10.0,
            daily_loss_limit_pct=5.0,
            current_daily_loss_pct=6.0,  # Exceeded limit
            reduce_only_mode=True,
            reduce_only_reason="Daily loss limit exceeded",
        )

        result = updater.compare_risk(old_risk, new_risk)

        assert result.has_changes
        assert "reduce_only_mode" in result.changed_fields

    def test_compare_system_no_changes(self):
        """Test system comparison with identical data."""
        updater = StateDeltaUpdater()
        system = SystemStatus(
            api_latency=0.05,
            connection_status="CONNECTED",
            rate_limit_usage="50%",
            memory_usage="100MB",
            cpu_usage="10%",
        )

        result = updater.compare_system(system, system)

        assert not result.has_changes

    def test_compare_system_connection_change(self):
        """Test system comparison detects connection status change."""
        updater = StateDeltaUpdater()
        old_system = SystemStatus(
            api_latency=0.05,
            connection_status="CONNECTED",
        )
        new_system = SystemStatus(
            api_latency=0.05,
            connection_status="DISCONNECTED",
        )

        result = updater.compare_system(old_system, new_system)

        assert result.has_changes
        assert "connection_status" in result.changed_fields
