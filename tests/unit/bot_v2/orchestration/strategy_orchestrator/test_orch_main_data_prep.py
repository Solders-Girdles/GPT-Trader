"""Tests for StrategyOrchestrator data preparation methods.

This module tests:
- Balance fetching and validation
- Equity extraction (USD/USDC)
- Position state building
- Mark price window management
- Equity adjustment for positions
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest


class TestEnsureBalances:
    """Test _ensure_balances async method."""

    @pytest.mark.asyncio
    async def test_returns_provided_balances(self, orchestrator, test_balance):
        """Test returns balances when provided."""
        balances = [test_balance]

        result = await orchestrator._ensure_balances(balances)

        assert result == balances

    @pytest.mark.asyncio
    async def test_fetches_balances_when_none(self, orchestrator, mock_bot, test_balance):
        """Test fetches balances from broker when None."""
        mock_bot.broker.list_balances = Mock(return_value=[test_balance])

        result = await orchestrator._ensure_balances(None)

        assert result == [test_balance]
        mock_bot.broker.list_balances.assert_called_once()


class TestExtractEquity:
    """Test _extract_equity method."""

    def test_extracts_usdc_balance(self, orchestrator):
        """Test extracts USDC balance."""
        balances = [
            Mock(asset="BTC", total=Decimal("1")),
            Mock(asset="USDC", total=Decimal("10000")),
        ]

        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("10000")

    def test_extracts_usd_balance(self, orchestrator):
        """Test extracts USD balance."""
        balances = [
            Mock(asset="BTC", total=Decimal("1")),
            Mock(asset="USD", total=Decimal("5000")),
        ]

        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("5000")

    def test_returns_zero_when_no_usd_balance(self, orchestrator):
        """Test returns zero when no USD/USDC balance found."""
        balances = [
            Mock(asset="BTC", total=Decimal("1")),
            Mock(asset="ETH", total=Decimal("10")),
        ]

        equity = orchestrator._extract_equity(balances)

        assert equity == Decimal("0")


class TestEnsurePositions:
    """Test _ensure_positions async method."""

    @pytest.mark.asyncio
    async def test_returns_provided_position_map(self, orchestrator, test_position):
        """Test returns position map when provided."""
        position_map = {"BTC-PERP": test_position}

        result = await orchestrator._ensure_positions(position_map)

        assert result == position_map

    @pytest.mark.asyncio
    async def test_fetches_positions_when_none(self, orchestrator, mock_bot, test_position):
        """Test fetches positions from broker when None."""
        mock_bot.broker.list_positions = Mock(return_value=[test_position])

        result = await orchestrator._ensure_positions(None)

        assert result == {"BTC-PERP": test_position}
        mock_bot.broker.list_positions.assert_called_once()


class TestBuildPositionState:
    """Test _build_position_state method."""

    def test_returns_none_when_no_position(self, orchestrator):
        """Test returns None state when symbol not in positions."""
        positions = {}

        state, quantity = orchestrator._build_position_state("BTC-PERP", positions)

        assert state is None
        assert quantity == Decimal("0")

    def test_builds_state_from_position(self, orchestrator, test_position):
        """Test builds position state from Position object."""
        positions = {"BTC-PERP": test_position}

        state, quantity = orchestrator._build_position_state("BTC-PERP", positions)

        assert state is not None
        assert state["quantity"] == Decimal("0.5")
        assert state["side"] == "long"
        assert state["entry"] == Decimal("50000")
        assert quantity == Decimal("0.5")


class TestGetMarks:
    """Test _get_marks method."""

    def test_returns_marks_for_symbol(self, orchestrator, mock_bot):
        """Test returns marks from bot mark_windows."""
        state = mock_bot.runtime_state
        state.mark_windows.clear()
        state.mark_windows["BTC-PERP"] = [Decimal("50000"), Decimal("51000")]

        marks = orchestrator._get_marks("BTC-PERP")

        assert marks == [Decimal("50000"), Decimal("51000")]

    def test_returns_empty_list_when_no_marks(self, orchestrator, mock_bot):
        """Test returns empty list when no marks for symbol."""
        mock_bot.runtime_state.mark_windows.clear()

        marks = orchestrator._get_marks("BTC-PERP")

        assert marks == []


class TestAdjustEquity:
    """Test _adjust_equity method."""

    def test_adjusts_equity_for_position(self, orchestrator):
        """Test adds position value to equity."""
        equity = Decimal("10000")
        position_quantity = Decimal("0.5")
        marks = [Decimal("50000"), Decimal("51000")]

        adjusted = orchestrator._adjust_equity(equity, position_quantity, marks, "BTC-PERP")

        # 10000 + (0.5 * 51000) = 10000 + 25500 = 35500
        assert adjusted == Decimal("35500")

    def test_returns_original_when_no_position(self, orchestrator):
        """Test returns original equity when no position."""
        equity = Decimal("10000")
        marks = [Decimal("50000")]

        adjusted = orchestrator._adjust_equity(equity, Decimal("0"), marks, "BTC-PERP")

        assert adjusted == Decimal("10000")
