"""
Tests for runtime state collection, equity calculation, and broker failure scenarios.
"""

from decimal import Decimal
from unittest.mock import patch

import pytest


class TestRuntimeState:
    """Test runtime state collection and broker interactions."""

    @pytest.mark.asyncio
    async def test_collect_runtime_guard_state_success(
        self, guard_manager, fake_broker, fake_balance, fake_position
    ):
        """Test collect_runtime_guard_state collects data successfully."""
        # Mock the sync calls to return the expected values
        fake_broker.list_balances.return_value = [fake_balance]
        fake_broker.list_positions.return_value = [fake_position]

        # Patch the broker methods to return the mock values
        with patch.object(
            guard_manager,
            "_calculate_equity",
            return_value=(Decimal("9500"), [fake_balance], Decimal("9500")),
        ):
            fake_broker.get_position_pnl.return_value = {
                "realized_pnl": "100",
                "unrealized_pnl": "500",
            }

            state = guard_manager.collect_runtime_guard_state()

            assert state.equity == Decimal("9500")
            assert len(state.balances) == 1
            assert len(state.positions) == 1
            assert "BTC-PERP" in state.positions_pnl
            assert "BTC-PERP" in state.positions_dict

    @pytest.mark.asyncio
    async def test_collect_runtime_guard_state_broker_balance_failure(
        self, guard_manager, fake_broker
    ):
        """Test collect_runtime_guard_state handles broker balance failures."""
        fake_broker.list_balances.side_effect = Exception("Balance fetch failed")

        with pytest.raises(Exception):  # Should propagate broker exception
            guard_manager.collect_runtime_guard_state()

    @pytest.mark.asyncio
    async def test_collect_runtime_guard_state_broker_position_failure(
        self, guard_manager, fake_broker, fake_balance
    ):
        """Test collect_runtime_guard_state handles broker position failures."""
        fake_broker.list_balances.return_value = [fake_balance]
        fake_broker.list_positions.side_effect = Exception("Position fetch failed")

        with pytest.raises(Exception):  # Should propagate broker exception
            guard_manager.collect_runtime_guard_state()

    @pytest.mark.asyncio
    async def test_collect_runtime_guard_state_pnl_calculation_fallback(
        self, guard_manager, fake_broker, fake_balance, fake_position
    ):
        """Test collect_runtime_guard_state falls back to manual PnL calculation."""
        # Mock the sync calls to return the expected values
        fake_broker.list_balances.return_value = [fake_balance]
        fake_broker.list_positions.return_value = [fake_position]

        # Patch the broker methods to return the mock values
        with patch.object(
            guard_manager,
            "_calculate_equity",
            return_value=(Decimal("9500"), [fake_balance], Decimal("9500")),
        ):
            fake_broker.get_position_pnl.side_effect = Exception("PnL fetch failed")

            state = guard_manager.collect_runtime_guard_state()

            # Should have calculated unrealized PnL manually
            assert "BTC-PERP" in state.positions_pnl
            assert state.positions_pnl["BTC-PERP"]["unrealized_pnl"] == Decimal(
                "500"
            )  # (51000 - 50000) * 0.5
